// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Internal real Cholesky computation over normalized usize descriptors.
//! Preconditions: f32/f64, valid square factor geometry/capacity, matching RHS
//! rows, valid borrowed addresses, disjoint factor/RHS spans, and selected
//! triangle interpretation supplied by the frontend. No API adapter is imported.
//! No full-input finite scans run here. The default preserves the checked
//! algorithm; an explicit internal policy permits POTRF-style IEEE propagation.
//! These routines alone do not establish a standard API or vendor ABI profile.

const std = @import("std");
const dense = @import("../../numerics/dense.zig");
const update = @import("../column_update.zig");
pub const Triangle = enum { upper, lower };
pub const FactorContract = @import("cholesky_contract.zig").FactorContract;
pub const SolveContract = @import("cholesky_contract.zig").SolveContract;
const SolveResult = @import("cholesky_contract.zig").SolveResult;
const substitute = @import("triangular_substitute.zig");

fn triangleIndex(row: usize, column: usize, ld: usize, triangle: Triangle) usize {
    return if (triangle == .lower) row + column * ld else column + row * ld;
}

/// Zero on success; otherwise the existing one-based numerical failure step.
/// Only the selected triangle is used; opposite triangle and padding survive.
pub fn factorize(comptime T: type, a: dense.Matrix(T), comptime triangle: Triangle) usize {
    return factorizeWithContract(T, a, triangle, .checked_v1);
}

/// The standard numerical policy rejects nonpositive/NaN pivots, permits
/// positive infinity, and stores off-diagonal IEEE results before continuing.
/// Neither policy adds a rejected-pivot store. The own unblocked update order
/// determines partial results; no recursive/blocked reference tail is promised.
/// Geometry and borrowed storage preconditions are identical to factorize.
pub fn factorizeWithContract(comptime T: type, a: dense.Matrix(T), comptime triangle: Triangle, comptime contract: FactorContract) usize {
    const n: usize = a.row_count;
    if (n == 0) return 0;
    const ld: usize = a.leading_dimension;
    for (0..n) |j| {
        var diagonal = a.values[j + j * ld];
        for (0..j) |k| {
            const entry = a.values[triangleIndex(j, k, ld, triangle)];
            diagonal -= entry * entry;
        }
        if (comptime contract == .checked_v1) {
            if (!(diagonal > 0) or !std.math.isFinite(diagonal)) return j + 1;
        } else {
            if (diagonal <= 0 or std.math.isNan(diagonal)) return j + 1;
        }
        const root = @sqrt(diagonal);
        a.values[j + j * ld] = root;
        if (triangle == .lower) {
            // Traverse each source and destination column contiguously. The
            // subtraction order for each entry remains k=0,...,j-1.
            const destination = a.values[j + 1 + j * ld .. n + j * ld];
            for (0..j) |k| update.subtractScaled(T, destination, a.values[j + 1 + k * ld .. n + k * ld], a.values[j + k * ld]);
            for (destination) |*entry| {
                const value = entry.* / root;
                if (comptime contract == .checked_v1) {
                    if (!std.math.isFinite(value)) return j + 1;
                }
                entry.* = value;
            }
            continue;
        }
        for (j + 1..n) |i| {
            const position = triangleIndex(i, j, ld, triangle);
            var entry = a.values[position];
            for (0..j) |k| {
                entry -= a.values[triangleIndex(i, k, ld, triangle)] *
                    a.values[triangleIndex(j, k, ld, triangle)];
            }
            entry /= root;
            if (comptime contract == .checked_v1) {
                if (!std.math.isFinite(entry)) return j + 1;
            }
            a.values[position] = entry;
        }
    }
    return 0;
}

/// Existing numerical diagonal and substitution policy, without frontend scans.
/// A numerical failure may retain a partial RHS; factor storage is read-only.
pub fn solve(comptime T: type, factor: dense.ConstMatrix(T), rhs: dense.Matrix(T), comptime triangle: Triangle) usize {
    return solveWithContract(T, factor, rhs, triangle, .checked_v1);
}

/// Internal non-checked policies use full diagonals and two all-RHS legs.
/// finite_v1 requires its finite numerical domain; ieee_v1 continues exceptional
/// arithmetic using Zynum's rules. Neither establishes standard POTRS API/ABI.
pub fn solveWithContract(comptime T: type, factor: dense.ConstMatrix(T), rhs: dense.Matrix(T), comptime triangle: Triangle, comptime contract: SolveContract) SolveResult(contract) {
    return substitute.solve(T, factor, rhs, switch (triangle) {
        .lower => .lower,
        .upper => .upper,
    }, contract);
}

test "real Cholesky core computes with valid native width strides and empty dimensions" {
    inline for (.{ f32, f64 }) |T| {
        inline for (.{ Triangle.lower, Triangle.upper }) |triangle| {
            var a = [_]T{4};
            var b = [_]T{12};
            const factor: dense.Matrix(T) = .{ .values = &a, .row_count = 1, .column_count = 1, .leading_dimension = std.math.maxInt(usize) };
            const rhs: dense.Matrix(T) = .{ .values = &b, .row_count = 1, .column_count = 1, .leading_dimension = std.math.maxInt(usize) };
            try std.testing.expectEqual(@as(usize, 1), try factor.check());
            try std.testing.expectEqual(@as(usize, 1), try rhs.check());
            try std.testing.expectEqual(@as(usize, 0), factorize(T, factor, triangle));
            try std.testing.expectEqual(@as(T, 2), a[0]);
            try std.testing.expectEqual(@as(usize, 0), solve(T, factor.asConst(), rhs, triangle));
            try std.testing.expectEqual(@as(T, 3), b[0]);
            const empty: dense.Matrix(T) = .{ .values = &.{}, .row_count = 0, .column_count = 0, .leading_dimension = 0 };
            const many_rhs: dense.Matrix(T) = .{ .values = &.{}, .row_count = 0, .column_count = @as(usize, std.math.maxInt(i32)) + 1, .leading_dimension = std.math.maxInt(usize) };
            try std.testing.expectEqual(@as(usize, 0), try many_rhs.check());
            try std.testing.expectEqual(@as(usize, 0), factorize(T, empty, triangle));
            try std.testing.expectEqual(@as(usize, 0), solve(T, empty.asConst(), many_rhs, triangle));
        }
    }
}

test "computational core has numerical failure policy without strict full input scan" {
    inline for (.{ f32, f64 }) |T| {
        var values = [_]T{ 4, 0, 77, std.math.nan(T) };
        const a: dense.Matrix(T) = .{ .values = &values, .row_count = 2, .column_count = 2, .leading_dimension = 2 };
        try std.testing.expectEqual(@as(usize, 4), try a.check());
        // A frontend which elects not to reject NaN upfront gets the retained
        // numerical step status and partial factor, not checked API Error.
        try std.testing.expectEqual(@as(usize, 2), factorize(T, a, .lower));
        try std.testing.expectEqual(@as(T, 2), values[0]);
        try std.testing.expectEqual(@as(T, 77), values[2]);
        try std.testing.expect(std.math.isNan(values[3]));
    }
}
