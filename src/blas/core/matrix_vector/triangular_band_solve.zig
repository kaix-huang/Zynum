// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Serial unit-stride triangular-band solve kernels.

const builtin = @import("builtin");

const scalar = @import("../shared/scalar.zig");
const indexing = @import("../shared/indexing.zig");
const vector_ops = @import("../vector/operations.zig");

const BlasInt = scalar.BlasInt;
const Order = scalar.Order;
const Uplo = scalar.Uplo;
const Diag = scalar.Diag;

fn isSupportedScalar(comptime T: type) bool {
    return T == f32 or T == f64 or T == scalar.ComplexF32 or T == scalar.ComplexF64;
}

const complex_vector_min = 64;
const complex_vector_parallel_min = 512 * 1024;

fn gateAllows(comptime T: type, require_x86: bool, n: BlasInt, k: BlasInt, incx: BlasInt) bool {
    if (comptime !isSupportedScalar(T)) return false;
    if (require_x86 and builtin.cpu.arch != .x86_64) return false;
    return incx == 1 and n >= 512 and k >= 0 and k <= @divTrunc(n, 16);
}

inline fn opValue(comptime T: type, comptime trans: Order, value: T) T {
    return if (trans == .conj_trans) scalar.conj(T, value) else value;
}

const BandColumn = struct {
    row0: usize,
    row1: usize,
    offset: usize,
    diag_offset: usize,
};

inline fn bandColumn(comptime uplo: Uplo, n: usize, k: usize, lda: BlasInt, column: usize) BandColumn {
    const column_offset = column * indexing.toUsize(lda);
    if (comptime uplo == .upper) {
        const row0 = column - @min(column, k);
        return .{
            .row0 = row0,
            .row1 = column + 1,
            .offset = column_offset + k + row0 - column,
            .diag_offset = column_offset + k,
        };
    }
    return .{
        .row0 = column,
        .row1 = column + 1 + @min(k, n - column - 1),
        .offset = column_offset,
        .diag_offset = column_offset,
    };
}

fn axpyContiguous(comptime T: type, n: usize, alpha: T, a: [*]const T, x: [*]T) void {
    if (n == 0 or scalar.isZero(T, alpha)) return;
    if (comptime T == f32 or T == f64) return vector_ops.axpyUnitReal(T, n, alpha, a, x);
    if (n >= complex_vector_min and n < complex_vector_parallel_min) {
        return vector_ops.axpy(T, @intCast(n), alpha, a, 1, x, 1);
    }
    for (0..n) |i| x[i] = scalar.add(T, x[i], scalar.mul(T, alpha, a[i]));
}

fn dotContiguous(comptime T: type, comptime trans: Order, n: usize, a: [*]const T, x: [*]const T) T {
    if (n == 0) return scalar.zero(T);
    if (comptime T == f32 or T == f64) return vector_ops.dotUnitReal(T, n, a, x);
    if (n >= complex_vector_min and n < complex_vector_parallel_min) {
        return vector_ops.dot(T, @intCast(n), a, 1, x, 1, trans == .conj_trans);
    }

    var sum = scalar.zero(T);
    for (0..n) |i| sum = scalar.add(T, sum, scalar.mul(T, opValue(T, trans, a[i]), x[i]));
    return sum;
}

noinline fn solveBandWindowLeaf(
    comptime T: type,
    comptime uplo: Uplo,
    comptime trans: Order,
    comptime diag: Diag,
    n: usize,
    k: usize,
    a: [*]const T,
    lda: BlasInt,
    x: [*]T,
) void {
    if (comptime trans == .no_trans) {
        if (comptime uplo == .upper) {
            var column = n;
            while (column > 0) {
                column -= 1;
                const bounds = bandColumn(.upper, n, k, lda, column);
                var xj = x[column];
                if (comptime diag == .non_unit) xj = scalar.divv(T, xj, a[bounds.diag_offset]);
                x[column] = xj;
                axpyContiguous(T, column - bounds.row0, scalar.neg(T, xj), a + bounds.offset, x + bounds.row0);
            }
        } else {
            for (0..n) |column| {
                const bounds = bandColumn(.lower, n, k, lda, column);
                var xj = x[column];
                if (comptime diag == .non_unit) xj = scalar.divv(T, xj, a[bounds.diag_offset]);
                x[column] = xj;
                const row0 = column + 1;
                axpyContiguous(T, bounds.row1 - row0, scalar.neg(T, xj), a + bounds.offset + 1, x + row0);
            }
        }
        return;
    }

    if (comptime uplo == .upper) {
        for (0..n) |column| {
            const bounds = bandColumn(.upper, n, k, lda, column);
            var xj = scalar.sub(T, x[column], dotContiguous(T, trans, column - bounds.row0, a + bounds.offset, x + bounds.row0));
            if (comptime diag == .non_unit) xj = scalar.divv(T, xj, opValue(T, trans, a[bounds.diag_offset]));
            x[column] = xj;
        }
    } else {
        var column = n;
        while (column > 0) {
            column -= 1;
            const bounds = bandColumn(.lower, n, k, lda, column);
            const row0 = column + 1;
            var xj = scalar.sub(T, x[column], dotContiguous(T, trans, bounds.row1 - row0, a + bounds.offset + 1, x + row0));
            if (comptime diag == .non_unit) xj = scalar.divv(T, xj, opValue(T, trans, a[bounds.diag_offset]));
            x[column] = xj;
        }
    }
}

noinline fn solveBandWindowDispatch(
    comptime T: type,
    uplo: Uplo,
    trans: Order,
    diag: Diag,
    n: usize,
    k: usize,
    a: [*]const T,
    lda: BlasInt,
    x: [*]T,
) void {
    switch (uplo) {
        .upper => switch (trans) {
            .no_trans => switch (diag) {
                .unit => solveBandWindowLeaf(T, .upper, .no_trans, .unit, n, k, a, lda, x),
                .non_unit => solveBandWindowLeaf(T, .upper, .no_trans, .non_unit, n, k, a, lda, x),
            },
            .trans => switch (diag) {
                .unit => solveBandWindowLeaf(T, .upper, .trans, .unit, n, k, a, lda, x),
                .non_unit => solveBandWindowLeaf(T, .upper, .trans, .non_unit, n, k, a, lda, x),
            },
            .conj_trans => switch (diag) {
                .unit => solveBandWindowLeaf(T, .upper, .conj_trans, .unit, n, k, a, lda, x),
                .non_unit => solveBandWindowLeaf(T, .upper, .conj_trans, .non_unit, n, k, a, lda, x),
            },
        },
        .lower => switch (trans) {
            .no_trans => switch (diag) {
                .unit => solveBandWindowLeaf(T, .lower, .no_trans, .unit, n, k, a, lda, x),
                .non_unit => solveBandWindowLeaf(T, .lower, .no_trans, .non_unit, n, k, a, lda, x),
            },
            .trans => switch (diag) {
                .unit => solveBandWindowLeaf(T, .lower, .trans, .unit, n, k, a, lda, x),
                .non_unit => solveBandWindowLeaf(T, .lower, .trans, .non_unit, n, k, a, lda, x),
            },
            .conj_trans => switch (diag) {
                .unit => solveBandWindowLeaf(T, .lower, .conj_trans, .unit, n, k, a, lda, x),
                .non_unit => solveBandWindowLeaf(T, .lower, .conj_trans, .non_unit, n, k, a, lda, x),
            },
        },
    }
}

fn tryTbsvWithGate(
    comptime T: type,
    require_x86: bool,
    uplo: Uplo,
    trans: Order,
    diag: Diag,
    n: BlasInt,
    k: BlasInt,
    a: [*]const T,
    lda: BlasInt,
    x: [*]T,
    incx: BlasInt,
) bool {
    if (comptime !isSupportedScalar(T)) return false;
    if (!gateAllows(T, require_x86, n, k, incx)) return false;
    solveBandWindowDispatch(T, uplo, trans, diag, indexing.toUsize(n), indexing.toUsize(k), a, lda, x);
    return true;
}

pub noinline fn tryTbsv(
    comptime T: type,
    uplo: Uplo,
    trans: Order,
    diag: Diag,
    n: BlasInt,
    k: BlasInt,
    a: [*]const T,
    lda: BlasInt,
    x: [*]T,
    incx: BlasInt,
) bool {
    return tryTbsvWithGate(T, true, uplo, trans, diag, n, k, a, lda, x, incx);
}

pub const testing = if (builtin.is_test) struct {
    pub fn gateAllowsForX86(comptime T: type, n: BlasInt, k: BlasInt, incx: BlasInt) bool {
        return gateAllows(T, false, n, k, incx);
    }

    pub fn productionGateAllows(comptime T: type, n: BlasInt, k: BlasInt, incx: BlasInt) bool {
        return gateAllows(T, true, n, k, incx);
    }

    pub fn tryTbsvForX86(
        comptime T: type,
        uplo: Uplo,
        trans: Order,
        diag: Diag,
        n: BlasInt,
        k: BlasInt,
        a: [*]const T,
        lda: BlasInt,
        x: [*]T,
        incx: BlasInt,
    ) bool {
        return tryTbsvWithGate(T, false, uplo, trans, diag, n, k, a, lda, x, incx);
    }
} else struct {};
