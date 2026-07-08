// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const scalar = @import("../shared/scalar.zig");
const indexing = @import("../shared/indexing.zig");
const matrix_vector_ops = @import("../matrix_vector.zig");

pub const BlasInt = scalar.BlasInt;
pub const Order = scalar.Order;
pub const Uplo = scalar.Uplo;
pub const Side = scalar.Side;

const zero = scalar.zero;
const add = scalar.add;
const mul = scalar.mul;
const conj = scalar.conj;
const isComplex = scalar.isComplex;
const isZero = scalar.isZero;

const toUsize = indexing.toUsize;
const matIndex = indexing.matIndex;
const matrixValue = matrix_vector_ops.matrixValue;
const symValue = matrix_vector_ops.symValue;

pub fn symm(comptime T: type, side: Side, uplo: Uplo, m_: BlasInt, n_: BlasInt, alpha: T, a: [*]const T, lda: BlasInt, b: [*]const T, ldb: BlasInt, beta: T, c: [*]T, ldc: BlasInt, herm: bool) void {
    if (m_ <= 0 or n_ <= 0) return;
    const m = toUsize(m_);
    const n = toUsize(n_);
    for (0..n) |j| for (0..m) |i| {
        var sum = zero(T);
        if (side == .left) {
            for (0..m) |p| sum = add(T, sum, mul(T, symValue(T, uplo, a, lda, i, p, herm), b[matIndex(ldb, p, j)]));
        } else {
            for (0..n) |p| sum = add(T, sum, mul(T, b[matIndex(ldb, i, p)], symValue(T, uplo, a, lda, p, j, herm)));
        }
        const idxc = matIndex(ldc, i, j);
        c[idxc] = add(T, mul(T, alpha, sum), if (isZero(T, beta)) zero(T) else mul(T, beta, c[idxc]));
    };
}

pub fn syrk(comptime T: type, uplo: Uplo, trans_: Order, n_: BlasInt, k_: BlasInt, alpha: T, a: [*]const T, lda: BlasInt, beta: T, c: [*]T, ldc: BlasInt, herm: bool) void {
    if (n_ <= 0) return;
    const n = toUsize(n_);
    const k = toUsize(k_);
    for (0..n) |j| {
        const row0: usize = if (uplo == .upper) 0 else j;
        const row1: usize = if (uplo == .upper) j + 1 else n;
        for (row0..row1) |i| {
            var sum = zero(T);
            for (0..k) |p| {
                const ai = if (trans_ == .no_trans) a[matIndex(lda, i, p)] else matrixValue(T, trans_, a, lda, i, p);
                var aj = if (trans_ == .no_trans) a[matIndex(lda, j, p)] else matrixValue(T, trans_, a, lda, j, p);
                if (herm) aj = conj(T, aj);
                sum = add(T, sum, mul(T, ai, aj));
            }
            const idxc = matIndex(ldc, i, j);
            c[idxc] = add(T, mul(T, alpha, sum), if (isZero(T, beta)) zero(T) else mul(T, beta, c[idxc]));
            if (herm and i == j) {
                if (comptime isComplex(T)) c[idxc].im = 0;
            }
        }
    }
}

pub fn syr2k(comptime T: type, uplo: Uplo, trans_: Order, n_: BlasInt, k_: BlasInt, alpha: T, a: [*]const T, lda: BlasInt, b: [*]const T, ldb: BlasInt, beta: T, c: [*]T, ldc: BlasInt, herm: bool) void {
    if (n_ <= 0) return;
    const n = toUsize(n_);
    const k = toUsize(k_);
    for (0..n) |j| {
        const row0: usize = if (uplo == .upper) 0 else j;
        const row1: usize = if (uplo == .upper) j + 1 else n;
        for (row0..row1) |i| {
            var sum = zero(T);
            for (0..k) |p| {
                const ai = matrixValue(T, trans_, a, lda, i, p);
                const bi = matrixValue(T, trans_, b, ldb, i, p);
                var aj = matrixValue(T, trans_, a, lda, j, p);
                var bj = matrixValue(T, trans_, b, ldb, j, p);
                if (herm) {
                    aj = conj(T, aj);
                    bj = conj(T, bj);
                    sum = add(T, sum, add(T, mul(T, alpha, mul(T, ai, bj)), mul(T, conj(T, alpha), mul(T, bi, aj))));
                } else {
                    sum = add(T, sum, add(T, mul(T, ai, bj), mul(T, bi, aj)));
                }
            }
            const idxc = matIndex(ldc, i, j);
            const prod = if (herm) sum else mul(T, alpha, sum);
            c[idxc] = add(T, prod, if (isZero(T, beta)) zero(T) else mul(T, beta, c[idxc]));
            if (herm and i == j) {
                if (comptime isComplex(T)) c[idxc].im = 0;
            }
        }
    }
}
