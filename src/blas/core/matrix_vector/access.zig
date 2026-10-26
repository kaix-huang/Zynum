// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Shared storage access helpers for BLAS Level 2 and Level 3 kernels.

const scalar = @import("../shared/scalar.zig");
const indexing = @import("../shared/indexing.zig");

const BlasInt = scalar.BlasInt;
const Order = scalar.Order;
const Uplo = scalar.Uplo;
const Diag = scalar.Diag;

const one = scalar.one;
const zero = scalar.zero;
const conj = scalar.conj;
const isComplex = scalar.isComplex;

const matIndex = indexing.matIndex;
const packedIndex = indexing.packedIndex;
const triPackedIndex = indexing.triPackedIndex;

/// Physical column-oriented storage; a row-oriented Hermitian input is
/// represented by the opposite triangle and conjugate_matrix=true.
pub const HermitianStorage = enum { dense, band, packed_storage };

/// Borrowed numeric view with no ownership, allocation, or validation cache.
/// The caller validates dimensions/strides and keeps the storage alive. Reading
/// the diagonal accesses only its real component; its imaginary bits are ignored.
pub fn HermitianView(comptime T: type, comptime storage: HermitianStorage) type {
    if (!isComplex(T)) @compileError("Hermitian view requires a BLAS complex scalar");
    return struct {
        triangle: Uplo,
        order: usize,
        bandwidth: usize,
        data: [*]const T,
        leading_dimension: BlasInt,
        conjugate_matrix: bool,

        pub fn value(self: @This(), row: usize, col: usize) T {
            const direct = (self.triangle == .upper and row <= col) or
                (self.triangle == .lower and row >= col);
            const index: ?usize = switch (storage) {
                .dense => if (direct) matIndex(self.leading_dimension, row, col) else matIndex(self.leading_dimension, col, row),
                .band => indexing.symBandIndex(self.triangle, self.order, self.bandwidth, self.leading_dimension, row, col),
                .packed_storage => packedIndex(self.triangle, self.order, row, col),
            };
            const stored = index orelse return zero(T);
            if (row == col) return .{ .re = self.data[stored].re, .im = 0 };
            const result = self.data[stored];
            // Reflection conjugates once; a conjugated matrix view toggles it.
            return if ((!direct) != self.conjugate_matrix) conj(T, result) else result;
        }
    };
}

pub fn matrixValue(comptime T: type, trans: Order, a: [*]const T, lda: BlasInt, row: usize, col: usize) T {
    return switch (trans) {
        .no_trans => a[matIndex(lda, row, col)],
        .trans => a[matIndex(lda, col, row)],
        .conj_trans => conj(T, a[matIndex(lda, col, row)]),
    };
}

pub fn symValue(comptime T: type, uplo: Uplo, a: [*]const T, lda: BlasInt, row: usize, col: usize, herm: bool) T {
    if ((uplo == .upper and row <= col) or (uplo == .lower and row >= col)) {
        var v = a[matIndex(lda, row, col)];
        if (herm and row == col) {
            if (comptime isComplex(T)) v.im = 0;
        }
        return v;
    }
    var v = a[matIndex(lda, col, row)];
    if (herm) v = conj(T, v);
    return v;
}

pub fn symPackedValue(comptime T: type, uplo: Uplo, n: usize, ap: [*]const T, row: usize, col: usize, herm: bool) T {
    var v = ap[packedIndex(uplo, n, row, col)];
    const stored_direct = (uplo == .upper and row <= col) or (uplo == .lower and row >= col);
    if (herm and !stored_direct) v = conj(T, v);
    if (herm and row == col) {
        if (comptime isComplex(T)) v.im = 0;
    }
    return v;
}

pub fn triValue(comptime T: type, uplo: Uplo, diag: Diag, trans: Order, a: [*]const T, lda: BlasInt, row: usize, col: usize) T {
    const ar = if (trans == .no_trans) row else col;
    const ac = if (trans == .no_trans) col else row;
    if (ar == ac and diag == .unit) return one(T);
    if (uplo == .upper and ar > ac) return zero(T);
    if (uplo == .lower and ar < ac) return zero(T);
    var v = a[matIndex(lda, ar, ac)];
    if (trans == .conj_trans) v = conj(T, v);
    return v;
}

pub fn triPackedValue(comptime T: type, uplo: Uplo, diag: Diag, trans: Order, n: usize, ap: [*]const T, row: usize, col: usize) T {
    const ar = if (trans == .no_trans) row else col;
    const ac = if (trans == .no_trans) col else row;
    if (ar == ac and diag == .unit) return one(T);
    const idxp = triPackedIndex(uplo, n, ar, ac) orelse return zero(T);
    var v = ap[idxp];
    if (trans == .conj_trans) v = conj(T, v);
    return v;
}
