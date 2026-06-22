// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Shared storage access helpers for BLAS Level 2 and Level 3 kernels.

const scalar = @import("../scalar.zig");
const indexing = @import("../indexing.zig");

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
