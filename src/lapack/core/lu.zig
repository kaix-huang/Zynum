// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Native-width LU computation. Frontends own shape, storage, alias, finite and
//! pivot preflight; all operands are stable caller-owned borrows for this call.
//! Dimensions/strides and logical pivot rows are usize. The controlled generic
//! pivot writer stores each row immediately at the original algorithm's write
//! point; the reader returns validated zero-based rows. Neither may allocate,
//! fail, defer stores, or mutate anything beyond its selected output element.
//! No full pivot copy or scratch allocation is used. Numeric info is one-based
//! and can retain an earlier singularity; it is not the committed pivot count.
//! Separate existing real/complex arithmetic paths preserve their precision,
//! expression order and partial-store behavior. No vendor semantics are implied.
//! The closed IEEE policies continue exceptional arithmetic: GETRF reports only
//! a zero selected pivot, GETRS returns void. These are internal numeric policies,
//! not LAPACK symbols, argument validation, an ABI or vendor bit-equivalence.

const std = @import("std");
const dense = @import("../../numerics/dense.zig");
const arithmetic = @import("../../numerics/complex.zig");
const scalar = arithmetic.scalar;
const update = @import("../column_update.zig");
const triangular = @import("../../numerics/triangular_ieee.zig");

pub const FactorContract = enum { checked_v1, getrf_ieee_v1 };
pub const SolveContract = enum { checked_v1, getrs_ieee_v1 };
/// Pure LU mathematics, not a Context admission token or external ABI enum.
/// An eventual adapter explicitly decodes its own N/T/C representation.
pub const Operation = triangular.Operation;
pub fn SolveResult(comptime contract: SolveContract) type {
    return if (contract == .checked_v1) usize else void;
}

/// Preconditions include writer capacity and row-encoding representability.
pub fn factorize(comptime T: type, a: dense.Matrix(T), writer: anytype) usize {
    return factorizeWithContract(T, a, writer, .checked_v1);
}

/// No finite-input scan or storage preflight is added here. IEEE complex LU
/// retains widened abs1 and subtractProduct, then explicitly casts each updated
/// stored value. Its column quotients use divideIeee on the stored operands.
/// The checked specialization keeps every original finite check/store/return.
pub fn factorizeWithContract(comptime T: type, a: dense.Matrix(T), writer: anytype, comptime contract: FactorContract) usize {
    if (comptime scalar.isComplex(T)) return complexFactorize(T, a, writer, contract);
    if (T != f32 and T != f64) @compileError("LU core requires a supported real or complex scalar");
    return realFactorize(T, a, writer, contract);
}

/// Preconditions include a valid complete zero-based logical pivot sequence.
/// For nonempty solves zero-diagonal preflight still precedes RHS mutation.
pub fn solve(comptime T: type, factor: dense.ConstMatrix(T), reader: anytype, rhs: dense.Matrix(T)) usize {
    if (comptime scalar.isComplex(T)) return complexSolve(T, factor, reader, rhs);
    if (T != f32 and T != f64) @compileError("LU core requires a supported real or complex scalar");
    return realSolve(T, factor, reader, rhs);
}

/// All modes use the same borrowed factor/pivots. Checked remains normal-only;
/// the original solve entry and both of its numerical bodies are unchanged.
/// IEEE N uses forward swaps, unit-L then U; T/H uses U^T/H then unit-L^T/H
/// followed by reverse swaps. Each leg completes all RHS before the next leg.
pub fn solveWithContract(comptime T: type, factor: dense.ConstMatrix(T), reader: anytype, rhs: dense.Matrix(T), operation: Operation, comptime contract: SolveContract) SolveResult(contract) {
    if (comptime contract == .checked_v1) {
        if (operation != .normal) @panic("checked LU solve requires normal operation");
        return solve(T, factor, reader, rhs);
    }
    if (comptime T != f32 and T != f64 and !scalar.isComplex(T)) @compileError("LU core requires a supported real or complex scalar");
    solveIeee(T, factor, reader, rhs, operation);
}

fn realFactorize(comptime T: type, a: dense.Matrix(T), writer: anytype, comptime contract: FactorContract) usize {
    const rows: usize = a.row_count;
    const columns: usize = a.column_count;
    const steps = @min(rows, columns);
    if (steps == 0) return 0;
    const ld: usize = a.leading_dimension;
    var info: usize = 0;
    for (0..steps) |k| {
        var pivot = k;
        var magnitude = @abs(a.values[k + k * ld]);
        for (k + 1..rows) |i| {
            const candidate = @abs(a.values[i + k * ld]);
            if (candidate > magnitude) {
                magnitude = candidate;
                pivot = i;
            }
        }
        writer.writeUnchecked(k, pivot);
        if (comptime contract == .checked_v1) {
            if (!std.math.isFinite(magnitude)) return if (info == 0) k + 1 else info;
        }
        if (magnitude == 0) {
            if (info == 0) info = k + 1;
            continue;
        }
        if (pivot != k) {
            for (0..columns) |j| std.mem.swap(T, &a.values[k + j * ld], &a.values[pivot + j * ld]);
        }
        const diagonal = a.values[k + k * ld];
        for (k + 1..rows) |i| a.values[i + k * ld] /= diagonal;
        for (k + 1..columns) |j| {
            if (comptime contract == .checked_v1) {
                if (!update.subtractScaledFinite(T, a.values[k + 1 + j * ld .. rows + j * ld], a.values[k + 1 + k * ld .. rows + k * ld], a.values[k + j * ld])) return if (info == 0) k + 1 else info;
            } else {
                update.subtractScaled(T, a.values[k + 1 + j * ld .. rows + j * ld], a.values[k + 1 + k * ld .. rows + k * ld], a.values[k + j * ld]);
            }
        }
    }
    return info;
}

fn realSolve(comptime T: type, factor: dense.ConstMatrix(T), reader: anytype, rhs: dense.Matrix(T)) usize {
    const n: usize = factor.row_count;
    const columns: usize = rhs.column_count;
    if (n == 0 or columns == 0) return 0;
    const lda: usize = factor.leading_dimension;
    const ldb: usize = rhs.leading_dimension;
    for (0..n) |i| {
        if (factor.values[i + i * lda] == 0) return i + 1;
    }
    for (0..columns) |column| {
        const x = rhs.values[column * ldb .. column * ldb + n];
        for (0..n) |k| {
            const pivot: usize = reader.readUnchecked(k);
            if (pivot != k) std.mem.swap(T, &x[k], &x[pivot]);
        }
        for (0..n) |i| {
            const value = x[i];
            if (!std.math.isFinite(value)) return i + 1;
            update.subtractScaled(T, x[i + 1 ..], factor.values[i + 1 + i * lda .. n + i * lda], value);
        }
        var remaining = n;
        while (remaining > 0) {
            remaining -= 1;
            const i = remaining;
            var value = x[i];
            for (i + 1..n) |k| value -= factor.values[i + k * lda] * x[k];
            value /= factor.values[i + i * lda];
            if (!std.math.isFinite(value)) return i + 1;
            x[i] = value;
        }
    }
    return 0;
}

fn complexFactorize(comptime C: type, a: dense.Matrix(C), writer: anytype, comptime contract: FactorContract) usize {
    const m: usize = a.row_count;
    const n: usize = a.column_count;
    const steps = @min(m, n);
    if (steps == 0) return 0;
    const ld: usize = a.leading_dimension;
    var info: usize = 0;
    for (0..steps) |k| {
        var pivot = k;
        var largest = arithmetic.abs1(C, a.values[k + k * ld]);
        for (k + 1..m) |i| {
            const magnitude = arithmetic.abs1(C, a.values[i + k * ld]);
            if (magnitude > largest) {
                largest = magnitude;
                pivot = i;
            }
        }
        writer.writeUnchecked(k, pivot);
        if (largest == 0) {
            if (info == 0) info = k + 1;
            continue;
        }
        if (pivot != k) for (0..n) |j| std.mem.swap(C, &a.values[k + j * ld], &a.values[pivot + j * ld]);
        const diagonal = a.values[k + k * ld];
        for (k + 1..m) |i| {
            if (comptime contract == .checked_v1) {
                a.values[i + k * ld] = arithmetic.narrow(C, arithmetic.divide(C, arithmetic.widen(C, a.values[i + k * ld]), diagonal)) orelse return if (info == 0) k + 1 else info;
            } else {
                a.values[i + k * ld] = arithmetic.divideIeee(C, a.values[i + k * ld], diagonal);
            }
        }
        for (k + 1..n) |j| for (k + 1..m) |i| {
            var value = arithmetic.widen(C, a.values[i + j * ld]);
            arithmetic.subtractProduct(C, &value, a.values[i + k * ld], a.values[k + j * ld]);
            if (comptime contract == .checked_v1) {
                a.values[i + j * ld] = arithmetic.narrow(C, value) orelse return if (info == 0) k + 1 else info;
            } else {
                a.values[i + j * ld] = .{ .re = @floatCast(value.re), .im = @floatCast(value.im) };
            }
        };
    }
    return info;
}

fn complexSolve(comptime C: type, factor: dense.ConstMatrix(C), reader: anytype, rhs: dense.Matrix(C)) usize {
    const n: usize = factor.row_count;
    const nrhs: usize = rhs.column_count;
    if (n == 0 or nrhs == 0) return 0;
    const lda: usize = factor.leading_dimension;
    const ldb: usize = rhs.leading_dimension;
    for (0..n) |i| if (scalar.isZero(C, factor.values[i + i * lda])) return i + 1;
    for (0..nrhs) |column| {
        const x = rhs.values[column * ldb ..][0..n];
        for (0..n) |k| {
            const pivot: usize = reader.readUnchecked(k);
            if (pivot != k) std.mem.swap(C, &x[k], &x[pivot]);
        }
        for (0..n) |i| {
            var value = arithmetic.widen(C, x[i]);
            for (0..i) |k| arithmetic.subtractProduct(C, &value, factor.values[i + k * lda], x[k]);
            x[i] = arithmetic.narrow(C, value) orelse return i + 1;
        }
        var remaining = n;
        while (remaining != 0) {
            remaining -= 1;
            const i = remaining;
            var value = arithmetic.widen(C, x[i]);
            for (i + 1..n) |k| arithmetic.subtractProduct(C, &value, factor.values[i + k * lda], x[k]);
            x[i] = arithmetic.narrow(C, arithmetic.divide(C, value, factor.values[i + i * lda])) orelse return i + 1;
        }
    }
    return 0;
}

// Both LU legs borrow the common native-width triangular implementation.
// The unit-L leg does not access the U-owned diagonal or opposite triangle.
fn unitLowerIeee(comptime T: type, factor: dense.ConstMatrix(T), rhs: dense.Matrix(T), operation: Operation) void {
    triangular.solve(T, factor, rhs, .lower, operation, .unit);
}

fn upperIeee(comptime T: type, factor: dense.ConstMatrix(T), rhs: dense.Matrix(T), operation: Operation) void {
    triangular.solve(T, factor, rhs, .upper, operation, .non_unit);
}

fn applyPivots(comptime T: type, reader: anytype, rhs: dense.Matrix(T), reverse: bool) void {
    for (0..rhs.column_count) |column| {
        const x = rhs.values[column * rhs.leading_dimension ..][0..rhs.row_count];
        for (0..rhs.row_count) |step| {
            const k = if (reverse) rhs.row_count - 1 - step else step;
            const pivot: usize = reader.readUnchecked(k);
            if (pivot != k) std.mem.swap(T, &x[k], &x[pivot]);
        }
    }
}

fn solveIeee(comptime T: type, factor: dense.ConstMatrix(T), reader: anytype, rhs: dense.Matrix(T), operation: Operation) void {
    if (factor.row_count == 0 or rhs.column_count == 0) return;
    if (operation == .normal) {
        applyPivots(T, reader, rhs, false);
        unitLowerIeee(T, factor, rhs, operation);
        upperIeee(T, factor, rhs, operation);
    } else {
        upperIeee(T, factor, rhs, operation);
        unitLowerIeee(T, factor, rhs, operation);
        applyPivots(T, reader, rhs, true);
    }
}

test "LU IEEE isolated unit lower leg does not read the U-owned diagonal or upper triangle" {
    inline for (.{ f32, f64, scalar.ComplexF32, scalar.ComplexF64 }) |T| {
        const nan: T = if (comptime scalar.isComplex(T)) .{ .re = std.math.nan(scalar.Real(T)), .im = std.math.nan(scalar.Real(T)) } else std.math.nan(T);
        const factor = [_]T{ nan, scalar.zero(T), scalar.zero(T), nan, nan, scalar.zero(T), nan, nan, nan };
        for ([_]Operation{ .normal, .transposed, .adjoint }) |operation| {
            var rhs = [_]T{ scalar.one(T), scalar.one(T), scalar.one(T) };
            const before = rhs;
            unitLowerIeee(T, .{ .values = &factor, .row_count = 3, .column_count = 3, .leading_dimension = 3 }, .{ .values = &rhs, .row_count = 3, .column_count = 1, .leading_dimension = 3 }, operation);
            try std.testing.expectEqualSlices(u8, std.mem.asBytes(&before), std.mem.asBytes(&rhs));
        }
    }
}
