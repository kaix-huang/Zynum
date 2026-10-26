// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Shared native-width Cholesky substitutions. Caller owns complete geometry,
//! capacity, stable borrowed addresses and disjoint factor/RHS span validation.
//! Only selected factor entries and active RHS elements are accessed. No views,
//! adapter, allocation, cache, runtime configuration or kernel dispatch exists.
//! checked_v1 preserves the checked schedule/arithmetic/failing-store behavior.
//! potrs_finite_v1 requires finite operands, nonzero diagonal and representable
//! finite arithmetic intermediates/stores. potrs_ieee_v1 uses the same traversal
//! and continues exceptional arithmetic under Zynum's explicit quotient rules.
//! Neither policy is an API/ABI or vendor bit-equivalence claim.
const std = @import("std");
const dense = @import("../../numerics/dense.zig");
const arithmetic = @import("../../numerics/complex.zig");
const scalar = arithmetic.scalar;
const update = @import("../column_update.zig");
const contracts = @import("cholesky_contract.zig");
pub const Triangle = enum { upper, lower };
const Direction = enum { forward, backward };
const SolveContract = contracts.SolveContract;
const SolveResult = contracts.SolveResult;

fn success(comptime contract: SolveContract) SolveResult(contract) {
    return if (contract == .checked_v1) 0 else {};
}

fn virtualLower(comptime T: type, factor: dense.ConstMatrix(T), i: usize, k: usize, triangle: Triangle) T {
    const ld = factor.leading_dimension;
    const z = factor.values[if (triangle == .lower) i + k * ld else k + i * ld];
    return if (triangle == .upper) scalar.conj(T, z) else z;
}

fn Accumulator(comptime T: type, comptime contract: SolveContract) type {
    return if (contract == .checked_v1 and scalar.isComplex(T)) arithmetic.WideComplex(T) else T;
}

fn dotRow(comptime T: type, factor: dense.ConstMatrix(T), x: []T, triangle: Triangle, i: usize, comptime direction: Direction, comptime contract: SolveContract) SolveResult(contract) {
    const lda = factor.leading_dimension;
    var value: Accumulator(T, contract) = if (comptime contract == .checked_v1)
        (if (comptime scalar.isComplex(T)) arithmetic.widen(T, x[i]) else x[i])
    else
        scalar.mul(T, scalar.one(T), x[i]);
    const start: usize = if (direction == .forward) 0 else i + 1;
    const end: usize = if (direction == .forward) i else x.len;
    for (start..end) |k| {
        const coefficient = if (comptime contract == .checked_v1)
            (if (direction == .forward) virtualLower(T, factor, i, k, triangle) else scalar.conj(T, virtualLower(T, factor, k, i, triangle)))
        else
            scalar.conj(T, factor.values[k + i * lda]);
        if (comptime contract == .checked_v1 and scalar.isComplex(T)) {
            arithmetic.subtractProduct(T, &value, coefficient, x[k]);
        } else if (comptime scalar.isComplex(T)) {
            value = scalar.sub(T, value, scalar.mul(T, coefficient, x[k]));
        } else {
            value -= coefficient * x[k];
        }
    }
    if (comptime contract == .checked_v1 and scalar.isComplex(T)) {
        const diagonal: arithmetic.Wide(T) = factor.values[i + i * lda].re;
        value.re /= diagonal;
        value.im /= diagonal;
        x[i] = arithmetic.narrow(T, value) orelse return i + 1;
    } else if (comptime contract == .checked_v1) {
        value /= factor.values[i + i * lda];
        if (!std.math.isFinite(value)) return i + 1;
        x[i] = value;
    } else {
        x[i] = quotient(T, value, scalar.conj(T, factor.values[i + i * lda]), contract);
    }
    return success(contract);
}

fn dotColumn(comptime T: type, factor: dense.ConstMatrix(T), x: []T, triangle: Triangle, comptime direction: Direction, comptime contract: SolveContract) SolveResult(contract) {
    for (0..x.len) |step| {
        const i = if (direction == .forward) step else x.len - 1 - step;
        if (comptime contract == .checked_v1) {
            const info = dotRow(T, factor, x, triangle, i, direction, contract);
            if (info != 0) return info;
        } else {
            dotRow(T, factor, x, triangle, i, direction, contract);
        }
    }
    return success(contract);
}

fn quotient(comptime T: type, numerator: T, diagonal: T, comptime contract: SolveContract) T {
    if (comptime contract == .potrs_ieee_v1 and scalar.isComplex(T)) return arithmetic.divideIeee(T, numerator, diagonal);
    return finiteQuotient(T, numerator, diagonal);
}

fn finiteQuotient(comptime T: type, numerator: T, diagonal: T) T {
    if (comptime scalar.isComplex(T)) {
        const value = arithmetic.divide(T, arithmetic.widen(T, numerator), diagonal);
        return .{ .re = @floatCast(value.re), .im = @floatCast(value.im) };
    }
    return numerator / diagonal;
}

fn scatterColumn(comptime T: type, factor: dense.ConstMatrix(T), x: []T, triangle: Triangle, comptime contract: SolveContract) SolveResult(contract) {
    const n = x.len;
    const lda = factor.leading_dimension;
    for (0..n) |step| {
        const i = if (triangle == .lower) step else n - 1 - step;
        if (comptime contract == .checked_v1) {
            // Only real lower checked substitution uses scatter. Preserve the
            // exact original quotient, store, eight-lane update and tail.
            const value = x[i] / factor.values[i + i * lda];
            if (!std.math.isFinite(value)) return i + 1;
            x[i] = value;
            update.subtractScaled(T, x[i + 1 ..], factor.values[i + 1 + i * lda .. n + i * lda], value);
        } else {
            // Gate before division. A quotient that becomes zero still scatters.
            if (scalar.isZero(T, x[i])) continue;
            x[i] = quotient(T, x[i], factor.values[i + i * lda], contract);
            const start: usize = if (triangle == .lower) i + 1 else 0;
            const end: usize = if (triangle == .lower) n else i;
            for (start..end) |k| x[k] = scalar.sub(T, x[k], scalar.mul(T, x[i], factor.values[k + i * lda]));
        }
    }
    return success(contract);
}

/// Checked calls finish both legs of each RHS before the next RHS and preserve
/// the original positivity and finite-store status. Both other policies run the first
/// leg on every RHS, then the second; their result is void, never discarded info.
/// Empty computation returns before numerical reads for either policy.
pub fn solve(comptime T: type, factor: dense.ConstMatrix(T), rhs: dense.Matrix(T), triangle: Triangle, comptime contract: SolveContract) SolveResult(contract) {
    const n = factor.row_count;
    const columns = rhs.column_count;
    if (n == 0 or columns == 0) return success(contract);
    const lda = factor.leading_dimension;
    const ldb = rhs.leading_dimension;
    if (comptime contract == .checked_v1) {
        for (0..n) |i| {
            const diagonal = if (comptime scalar.isComplex(T)) factor.values[i + i * lda].re else factor.values[i + i * lda];
            if (!(diagonal > 0)) return i + 1;
        }
        for (0..columns) |column| {
            const x = rhs.values[column * ldb .. column * ldb + n];
            const first = if (comptime scalar.isComplex(T))
                dotColumn(T, factor, x, triangle, .forward, contract)
            else if (triangle == .lower)
                scatterColumn(T, factor, x, triangle, contract)
            else
                dotColumn(T, factor, x, triangle, .forward, contract);
            if (first != 0) return first;
            const second = dotColumn(T, factor, x, triangle, .backward, contract);
            if (second != 0) return second;
        }
    } else {
        for (0..columns) |column| {
            const x = rhs.values[column * ldb .. column * ldb + n];
            if (triangle == .lower) scatterColumn(T, factor, x, triangle, contract) else dotColumn(T, factor, x, triangle, .forward, contract);
        }
        for (0..columns) |column| {
            const x = rhs.values[column * ldb .. column * ldb + n];
            if (triangle == .lower) dotColumn(T, factor, x, triangle, .backward, contract) else scatterColumn(T, factor, x, triangle, contract);
        }
    }
    return success(contract);
}
