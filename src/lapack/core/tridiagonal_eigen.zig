// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Allocation-free implicit shifted QL for real symmetric tridiagonal matrices.
//! Frontends own argument/storage/alias checks and operation admission. Dimensions
//! and optional eigenvectors use the common native-width dense descriptor.
const std = @import("std");
const dense = @import("../../numerics/dense.zig");

pub const Contract = enum { checked_finite, stev_ieee_v1 };

pub const Status = enum { converged, iteration_limit, nonfinite_input, range_limit };
/// Native, zero-based half-open interval of the active unresolved block.
/// This is numerical outcome data; signed vendor INFO encoding belongs outside
/// the numerical core.
pub const UnconvergedBlock = struct { first: usize, limit: usize };
pub const Result = struct {
    status: Status,
    iterations: usize = 0,
    unconverged: usize = 0,
    failed_block: ?UnconvergedBlock = null,
};
pub const Options = struct { max_iterations: ?usize = null };

fn remaining(comptime T: type, e: []const T) usize {
    var count: usize = 0;
    for (e) |x| if (x != 0) {
        count += 1;
    };
    return count;
}

fn rotateColumns(comptime T: type, z: dense.Matrix(T), i: usize, c: f64, s: f64) void {
    for (0..z.row_count) |k| {
        const first = k + i * z.leading_dimension;
        const second = first + z.leading_dimension;
        const x: f64 = z.values[first];
        const y: f64 = z.values[second];
        z.values[first] = @floatCast(c * x - s * y);
        z.values[second] = @floatCast(s * x + c * y);
    }
}

fn powerScale(comptime T: type, value: T, exponent: i32) T {
    // Keep exact zero, including its sign, independent of exponent handling.
    return if (value == 0) value else std.math.ldexp(value, exponent);
}

fn restore(comptime T: type, d: []T, e: []T, exponent: i32) bool {
    var finite = true;
    for (d) |*x| {
        x.* = powerScale(T, x.*, exponent);
        finite = finite and std.math.isFinite(x.*);
    }
    for (e) |*x| {
        x.* = powerScale(T, x.*, exponent);
        finite = finite and std.math.isFinite(x.*);
    }
    return finite;
}

/// d and the first max(0,n-1) entries of e are overwritten. On success d ascends,
/// e is zero, and optional z holds matching orthonormal eigenvectors. z is n*n
/// column-major, initialized to identity; its padding is never written. null z
/// requires no vector storage. All used spans must be valid and disjoint.
///
/// Exact disconnected blocks are scaled separately by powers of two. Iterations
/// use f64 intermediates and round each updated d/e/vector entry to T. This is a
/// normwise solver; it does not promise relative accuracy for tiny eigenvalues.
/// Nonfinite input is rejected before mutation. On iteration_limit, d/e and z
/// retain the unfinished orthogonal iteration in original units; unconverged
/// counts nonzero off-diagonals. range_limit denotes unrepresentable rescaling,
/// distinct from ordinary nonconvergence. No allocation, execution admission,
/// adapter configuration, signed index model or vendor ABI is introduced here.
pub fn solve(comptime T: type, d: []T, off_diagonal: []T, vectors: ?dense.Matrix(T), options: Options) Result {
    return solveWithContract(T, d, off_diagonal, vectors, options, .checked_finite);
}

/// The standard driver policy permits IEEE nonfinite operands after its own
/// optional NaN scan. Iterations remain bounded and INFO derives only from
/// unconverged off-diagonals, not the checked API's finite-range status.
pub fn solveWithContract(comptime T: type, d: []T, off_diagonal: []T, vectors: ?dense.Matrix(T), options: Options, comptime contract: Contract) Result {
    if (T != f32 and T != f64) @compileError("tridiagonal eigen supports f32/f64");
    const n = d.len;
    const e = off_diagonal[0..n -| 1];
    if (comptime contract == .checked_finite) {
        for (d) |x| if (!std.math.isFinite(x)) return .{ .status = .nonfinite_input };
        for (e) |x| if (!std.math.isFinite(x)) return .{ .status = .nonfinite_input };
    }
    if (vectors) |z| for (0..n) |j| for (0..n) |i| {
        z.values[i + j * z.leading_dimension] = if (i == j) 1 else 0;
    };
    const limit = options.max_iterations orelse n *| 30;
    const eps: f64 = @as(f64, std.math.floatEps(T)) * 0.5;
    var iterations: usize = 0;
    var start: usize = 0;
    while (start < n) {
        var end = start;
        while (end + 1 < n and e[end] != 0) : (end += 1) {}
        if (end == start) {
            start += 1;
            continue;
        }
        var largest: T = 0;
        for (d[start .. end + 1]) |x| largest = @max(largest, @abs(x));
        for (e[start..end]) |x| largest = @max(largest, @abs(x));
        const exponent = std.math.frexp(largest).exponent;
        for (d[start .. end + 1]) |*x| x.* = powerScale(T, x.*, -exponent);
        for (e[start..end]) |*x| x.* = powerScale(T, x.*, -exponent);
        var l = start;
        while (l <= end) {
            var m = l;
            while (m < end) : (m += 1) {
                const threshold = eps * @sqrt(@abs(@as(f64, d[m]))) * @sqrt(@abs(@as(f64, d[m + 1])));
                if (@abs(@as(f64, e[m])) <= threshold) {
                    e[m] = 0;
                    break;
                }
            }
            if (m == l) {
                l += 1;
                continue;
            }
            if (iterations == limit) {
                const finite = restore(T, d[start .. end + 1], e[start..end], exponent);
                return .{
                    .status = if (finite or contract == .stev_ieee_v1) .iteration_limit else .range_limit,
                    .iterations = iterations,
                    .unconverged = remaining(T, e),
                    .failed_block = .{ .first = l, .limit = m + 1 },
                };
            }
            iterations += 1;
            const edge: f64 = e[l];
            const delta = 0.5 * (@as(f64, d[l + 1]) - @as(f64, d[l]));
            const radius = std.math.hypot(delta, edge);
            const denominator = delta + (if (delta < 0) -radius else radius);
            const shift_correction = edge * (edge / denominator);
            var g = @as(f64, d[m]) - @as(f64, d[l]) + shift_correction;
            var s: f64 = 1;
            var c: f64 = 1;
            var correction: f64 = 0;
            var i = m;
            var restart = false;
            while (i > l) {
                i -= 1;
                const f = s * @as(f64, e[i]);
                const b = c * @as(f64, e[i]);
                const radius_step = std.math.hypot(f, g);
                // The discarded bulge at m is not part of the n-1 input array.
                if (i + 1 < m) e[i + 1] = @floatCast(radius_step);
                if (radius_step == 0) {
                    d[i + 1] = @floatCast(@as(f64, d[i + 1]) - correction);
                    if (i + 1 < m) e[i + 1] = 0;
                    restart = true;
                    break;
                }
                s = f / radius_step;
                c = g / radius_step;
                g = @as(f64, d[i + 1]) - correction;
                const step = (@as(f64, d[i]) - g) * s + 2 * c * b;
                correction = s * step;
                d[i + 1] = @floatCast(g + correction);
                g = c * step - b;
                if (vectors) |z| rotateColumns(T, z, i, c, s);
            }
            if (restart) continue;
            d[l] = @floatCast(@as(f64, d[l]) - correction);
            e[l] = @floatCast(g);
        }
        const finite = restore(T, d[start .. end + 1], e[start..end], exponent);
        if (!finite and contract == .checked_finite) return .{ .status = .range_limit, .iterations = iterations, .unconverged = remaining(T, e) };
        start = end + 1;
    }
    for (0..n) |i| {
        var smallest = i;
        for (i + 1..n) |j| if (d[j] < d[smallest]) {
            smallest = j;
        };
        if (smallest != i) {
            std.mem.swap(T, &d[i], &d[smallest]);
            if (vectors) |z| for (0..n) |k| {
                std.mem.swap(T, &z.values[k + i * z.leading_dimension], &z.values[k + smallest * z.leading_dimension]);
            };
        }
    }
    return .{ .status = .converged, .iterations = iterations };
}
