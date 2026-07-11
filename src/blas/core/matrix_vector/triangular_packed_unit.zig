// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Unit-stride packed-column triangular matrix-vector and solve kernels.

const builtin = @import("builtin");

const scalar = @import("../shared/scalar.zig");
const vector_ops = @import("../vector/operations.zig");

const BlasInt = scalar.BlasInt;
const Order = scalar.Order;
const Uplo = scalar.Uplo;
const Diag = scalar.Diag;

const zero = scalar.zero;
const one = scalar.one;
const add = scalar.add;
const sub = scalar.sub;
const mul = scalar.mul;
const divv = scalar.divv;
const conj = scalar.conj;
const neg = scalar.neg;
const isComplex = scalar.isComplex;
const isZero = scalar.isZero;

const vector_min = 64;
const vector_parallel_min = 512 * 1024;
const production_min_n = 128;

const PackedColumn = struct {
    row0: usize,
    row1: usize,
    offset: usize,
    diag_offset: usize,
};

fn isSupportedType(comptime T: type) bool {
    return T == f32 or T == f64 or T == scalar.ComplexF32 or T == scalar.ComplexF64;
}

fn productionGateAllows(comptime T: type, n: usize, incx: BlasInt) bool {
    if (comptime builtin.cpu.arch != .x86_64 or !isSupportedType(T)) return false;
    return incx == 1 and n >= production_min_n;
}

fn packedColumn(comptime uplo: Uplo, n: usize, column: usize) PackedColumn {
    if (comptime uplo == .upper) {
        const offset = column * (column + 1) / 2;
        return .{
            .row0 = 0,
            .row1 = column + 1,
            .offset = offset,
            .diag_offset = offset + column,
        };
    }

    const offset = column * (2 * n - column + 1) / 2;
    return .{
        .row0 = column,
        .row1 = n,
        .offset = offset,
        .diag_offset = offset,
    };
}

fn packedAxpy(comptime T: type, n: usize, alpha: T, a: [*]const T, x: [*]T) void {
    if (n == 0 or isZero(T, alpha)) return;
    if (n >= vector_min) {
        if (comptime isComplex(T)) {
            // A triangular dependency step must finish before the next column.
            if (n < vector_parallel_min) return vector_ops.axpy(T, @intCast(n), alpha, a, 1, x, 1);
        } else {
            return vector_ops.axpyUnitReal(T, n, alpha, a, x);
        }
    }
    for (0..n) |i| x[i] = add(T, x[i], mul(T, alpha, a[i]));
}

fn packedDot(comptime T: type, comptime conjugate_a: bool, n: usize, a: [*]const T, x: [*]const T) T {
    if (n >= vector_min) {
        if (comptime isComplex(T)) {
            if (n < vector_parallel_min) return vector_ops.dot(T, @intCast(n), a, 1, x, 1, conjugate_a);
        } else {
            return vector_ops.dotUnitReal(T, n, a, x);
        }
    }

    var sum = zero(T);
    for (0..n) |i| {
        const av = if (comptime conjugate_a) conj(T, a[i]) else a[i];
        sum = add(T, sum, mul(T, av, x[i]));
    }
    return sum;
}

fn tpmvUnitLeaf(
    comptime T: type,
    comptime uplo: Uplo,
    comptime trans_: Order,
    comptime diag: Diag,
    n: usize,
    ap: [*]const T,
    x: [*]T,
) void {
    if (comptime trans_ == .no_trans) {
        if (comptime uplo == .upper) {
            for (0..n) |column_index| {
                const metadata = packedColumn(uplo, n, column_index);
                const xj = x[column_index];
                packedAxpy(T, column_index - metadata.row0, xj, ap + metadata.offset, x + metadata.row0);
                if (comptime diag == .non_unit) x[column_index] = mul(T, ap[metadata.diag_offset], xj);
            }
        } else {
            var column_index = n;
            while (column_index > 0) {
                column_index -= 1;
                const metadata = packedColumn(uplo, n, column_index);
                const xj = x[column_index];
                if (comptime diag == .non_unit) x[column_index] = mul(T, ap[metadata.diag_offset], xj);
                const row0 = column_index + 1;
                packedAxpy(T, metadata.row1 - row0, xj, ap + metadata.offset + 1, x + row0);
            }
        }
        return;
    }

    const conjugate_a = trans_ == .conj_trans;
    if (comptime uplo == .upper) {
        var column_index = n;
        while (column_index > 0) {
            column_index -= 1;
            const metadata = packedColumn(uplo, n, column_index);
            var sum = packedDot(T, conjugate_a, column_index - metadata.row0, ap + metadata.offset, x + metadata.row0);
            const diagonal = if (comptime diag == .unit)
                one(T)
            else if (comptime conjugate_a)
                conj(T, ap[metadata.diag_offset])
            else
                ap[metadata.diag_offset];
            sum = add(T, sum, mul(T, diagonal, x[column_index]));
            x[column_index] = sum;
        }
    } else {
        for (0..n) |column_index| {
            const metadata = packedColumn(uplo, n, column_index);
            const diagonal = if (comptime diag == .unit)
                one(T)
            else if (comptime conjugate_a)
                conj(T, ap[metadata.diag_offset])
            else
                ap[metadata.diag_offset];
            var sum = mul(T, diagonal, x[column_index]);
            const row0 = column_index + 1;
            sum = add(T, sum, packedDot(T, conjugate_a, metadata.row1 - row0, ap + metadata.offset + 1, x + row0));
            x[column_index] = sum;
        }
    }
}

fn tpsvUnitLeaf(
    comptime T: type,
    comptime uplo: Uplo,
    comptime trans_: Order,
    comptime diag: Diag,
    n: usize,
    ap: [*]const T,
    x: [*]T,
) void {
    if (comptime trans_ == .no_trans) {
        if (comptime uplo == .upper) {
            var column_index = n;
            while (column_index > 0) {
                column_index -= 1;
                const metadata = packedColumn(uplo, n, column_index);
                var xj = x[column_index];
                if (comptime diag == .non_unit) xj = divv(T, xj, ap[metadata.diag_offset]);
                x[column_index] = xj;
                packedAxpy(T, column_index - metadata.row0, neg(T, xj), ap + metadata.offset, x + metadata.row0);
            }
        } else {
            for (0..n) |column_index| {
                const metadata = packedColumn(uplo, n, column_index);
                var xj = x[column_index];
                if (comptime diag == .non_unit) xj = divv(T, xj, ap[metadata.diag_offset]);
                x[column_index] = xj;
                const row0 = column_index + 1;
                packedAxpy(T, metadata.row1 - row0, neg(T, xj), ap + metadata.offset + 1, x + row0);
            }
        }
        return;
    }

    const conjugate_a = trans_ == .conj_trans;
    if (comptime uplo == .upper) {
        for (0..n) |column_index| {
            const metadata = packedColumn(uplo, n, column_index);
            var xj = sub(T, x[column_index], packedDot(T, conjugate_a, column_index - metadata.row0, ap + metadata.offset, x + metadata.row0));
            if (comptime diag == .non_unit) {
                const diagonal = if (comptime conjugate_a) conj(T, ap[metadata.diag_offset]) else ap[metadata.diag_offset];
                xj = divv(T, xj, diagonal);
            }
            x[column_index] = xj;
        }
    } else {
        var column_index = n;
        while (column_index > 0) {
            column_index -= 1;
            const metadata = packedColumn(uplo, n, column_index);
            const row0 = column_index + 1;
            var xj = sub(T, x[column_index], packedDot(T, conjugate_a, metadata.row1 - row0, ap + metadata.offset + 1, x + row0));
            if (comptime diag == .non_unit) {
                const diagonal = if (comptime conjugate_a) conj(T, ap[metadata.diag_offset]) else ap[metadata.diag_offset];
                xj = divv(T, xj, diagonal);
            }
            x[column_index] = xj;
        }
    }
}

fn dispatchTpmv(comptime T: type, uplo: Uplo, trans_: Order, diag: Diag, n: usize, ap: [*]const T, x: [*]T) void {
    switch (uplo) {
        inline else => |comptime_uplo| switch (trans_) {
            inline else => |comptime_trans| switch (diag) {
                inline else => |comptime_diag| tpmvUnitLeaf(T, comptime_uplo, comptime_trans, comptime_diag, n, ap, x),
            },
        },
    }
}

fn dispatchTpsv(comptime T: type, uplo: Uplo, trans_: Order, diag: Diag, n: usize, ap: [*]const T, x: [*]T) void {
    switch (uplo) {
        inline else => |comptime_uplo| switch (trans_) {
            inline else => |comptime_trans| switch (diag) {
                inline else => |comptime_diag| tpsvUnitLeaf(T, comptime_uplo, comptime_trans, comptime_diag, n, ap, x),
            },
        },
    }
}

pub noinline fn tryTpmv(
    comptime T: type,
    uplo: Uplo,
    trans_: Order,
    diag: Diag,
    n: usize,
    ap: [*]const T,
    x: [*]T,
    incx: BlasInt,
) bool {
    if (!productionGateAllows(T, n, incx)) return false;
    dispatchTpmv(T, uplo, trans_, diag, n, ap, x);
    return true;
}

pub noinline fn tryTpsv(
    comptime T: type,
    uplo: Uplo,
    trans_: Order,
    diag: Diag,
    n: usize,
    ap: [*]const T,
    x: [*]T,
    incx: BlasInt,
) bool {
    if (!productionGateAllows(T, n, incx)) return false;
    dispatchTpsv(T, uplo, trans_, diag, n, ap, x);
    return true;
}

pub const testing = if (builtin.is_test) struct {
    pub fn gateAllows(comptime T: type, n: usize, incx: BlasInt) bool {
        return productionGateAllows(T, n, incx);
    }

    pub fn runTpmvUnit(comptime T: type, uplo: Uplo, trans_: Order, diag: Diag, n: usize, ap: [*]const T, x: [*]T) void {
        dispatchTpmv(T, uplo, trans_, diag, n, ap, x);
    }

    pub fn runTpsvUnit(comptime T: type, uplo: Uplo, trans_: Order, diag: Diag, n: usize, ap: [*]const T, x: [*]T) void {
        dispatchTpsv(T, uplo, trans_, diag, n, ap, x);
    }
} else struct {};
