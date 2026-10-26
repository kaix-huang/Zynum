// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Internal complex Cholesky computation over native-width dense operands.
//! Preconditions: existing ComplexF32/F64 identity, valid square factor geometry,
//! matching RHS rows, truthful stable borrowed buffers, disjoint factor/RHS spans,
//! and frontend-selected triangle interpretation. Frontends own argument and
//! input policies; these routines perform no upfront finite/diagonal-form scan.
//! Default checked behavior preserves its original widened arithmetic, stored
//! order, numeric status and partial mutation. Explicit lapack_potrf selects
//! pivot/IEEE rules within those same loops, not vendor or ABI equivalence.

const std = @import("std");
const dense = @import("../../numerics/dense.zig");
const arithmetic = @import("../../numerics/complex.zig");
const scalar = arithmetic.scalar;
pub const Triangle = enum { upper, lower };
pub const FactorContract = @import("cholesky_contract.zig").FactorContract;
pub const SolveContract = @import("cholesky_contract.zig").SolveContract;
const SolveResult = @import("cholesky_contract.zig").SolveResult;
const substitute = @import("triangular_substitute.zig");

fn triangleIndex(row: usize, column: usize, ld: usize, triangle: Triangle) usize {
    return if (triangle == .lower) row + column * ld else column + row * ld;
}
fn lower(comptime C: type, a: anytype, i: usize, j: usize, ld: usize, triangle: Triangle) C {
    const z = a.values[triangleIndex(i, j, ld, triangle)];
    return if (triangle == .upper) scalar.conj(C, z) else z;
}
fn storeLower(comptime C: type, a: dense.Matrix(C), i: usize, j: usize, ld: usize, triangle: Triangle, z: C) void {
    a.values[triangleIndex(i, j, ld, triangle)] = if (triangle == .upper) scalar.conj(C, z) else z;
}

/// Zero on success; otherwise the one-based failing factorization step. A
/// rejected pivot is not stored; a failed off-diagonal narrowing leaves that
/// entry intact after the diagonal and any earlier entries have been stored.
/// Only the selected triangle is accessed; padding/opposite triangle survive.
pub fn factorize(comptime C: type, a: dense.Matrix(C), triangle: Triangle) usize {
    return factorizeWithContract(C, a, triangle, .checked_v1);
}

/// lapack_potrf rejects nonpositive/NaN real pivot residues but permits positive
/// infinity and off-diagonal IEEE results. Diagonal imaginary input is ignored;
/// it becomes +0 only after an accepted pivot. Rejected pivots are not assigned.
/// Geometry and stable borrowed storage preconditions match factorize. Partial
/// results follow this unblocked order, not recursive POTRF2 or POTF2 tails.
pub fn factorizeWithContract(comptime C: type, a: dense.Matrix(C), triangle: Triangle, comptime contract: FactorContract) usize {
    const n: usize = a.row_count;
    if (n == 0) return 0;
    const ld: usize = a.leading_dimension;
    const W = arithmetic.Wide(C);
    for (0..n) |j| {
        var diagonal: W = a.values[j + j * ld].re;
        for (0..j) |k| {
            const z = arithmetic.widen(C, lower(C, a, j, k, ld, triangle));
            diagonal -= z.re * z.re + z.im * z.im;
        }
        if (comptime contract == .checked_v1) {
            if (!(diagonal > 0) or !std.math.isFinite(diagonal)) return j + 1;
        } else {
            if (diagonal <= 0 or std.math.isNan(diagonal)) return j + 1;
        }
        const root: scalar.Real(C) = @floatCast(@sqrt(diagonal));
        if (comptime contract == .checked_v1) {
            if (!(root > 0) or !std.math.isFinite(root)) return j + 1;
        }
        a.values[j + j * ld] = .{ .re = root, .im = 0 };
        for (j + 1..n) |i| {
            var value = arithmetic.widen(C, lower(C, a, i, j, ld, triangle));
            for (0..j) |k| arithmetic.subtractProduct(C, &value, lower(C, a, i, k, ld, triangle), scalar.conj(C, lower(C, a, j, k, ld, triangle)));
            value.re /= @as(W, root);
            value.im /= @as(W, root);
            // Preserve narrow's component-cast order; only its finite-result
            // admission differs between these two numerical contracts.
            const stored: C = .{ .re = @floatCast(value.re), .im = @floatCast(value.im) };
            if (comptime contract == .checked_v1) {
                if (!arithmetic.finite(C, stored)) return j + 1;
            }
            storeLower(C, a, i, j, ld, triangle, stored);
        }
    }
    return 0;
}

/// Positive-diagonal preflight precedes any RHS store for nonempty solves.
/// Arithmetic failure leaves completed substitutions and earlier RHS columns;
/// the failing assignment and all later columns remain unchanged. Empty systems
/// or zero RHS columns do not read factor entries or apply positivity checks.
pub fn solve(comptime C: type, factor: dense.ConstMatrix(C), rhs: dense.Matrix(C), triangle: Triangle) usize {
    return solveWithContract(C, factor, rhs, triangle, .checked_v1);
}

/// Internal non-checked policies use full diagonals and two all-RHS legs.
/// finite_v1 requires its finite numerical domain; ieee_v1 continues exceptional
/// arithmetic using Zynum's rules. Neither establishes standard POTRS API/ABI.
pub fn solveWithContract(comptime C: type, factor: dense.ConstMatrix(C), rhs: dense.Matrix(C), triangle: Triangle, comptime contract: SolveContract) SolveResult(contract) {
    return substitute.solve(C, factor, rhs, switch (triangle) {
        .lower => .lower,
        .upper => .upper,
    }, contract);
}
