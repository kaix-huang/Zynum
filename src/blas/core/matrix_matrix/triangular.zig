// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const scalar = @import("../shared/scalar.zig");
const indexing = @import("../shared/indexing.zig");
const matrix_vector_ops = @import("../matrix_vector.zig");

pub const BlasInt = scalar.BlasInt;
pub const Order = scalar.Order;
pub const Uplo = scalar.Uplo;
pub const Diag = scalar.Diag;
pub const Side = scalar.Side;

const zero = scalar.zero;
const add = scalar.add;
const mul = scalar.mul;
const isOne = scalar.isOne;

const toUsize = indexing.toUsize;
const matIndex = indexing.matIndex;
const triValue = matrix_vector_ops.triValue;
const trmv = matrix_vector_ops.trmv;
const trsv = matrix_vector_ops.trsv;

fn opIsUpper(uplo: Uplo, trans_: Order) bool {
    return (trans_ == .no_trans and uplo == .upper) or (trans_ != .no_trans and uplo == .lower);
}

fn scaleDenseMatrix(comptime T: type, m: usize, n: usize, alpha: T, b: [*]T, ldb: BlasInt) void {
    if (isOne(T, alpha)) return;
    for (0..n) |j| {
        for (0..m) |i| {
            b[matIndex(ldb, i, j)] = mul(T, alpha, b[matIndex(ldb, i, j)]);
        }
    }
}

pub fn trmm(comptime T: type, side: Side, uplo: Uplo, trans_: Order, diag: Diag, m_: BlasInt, n_: BlasInt, alpha: T, a: [*]const T, lda: BlasInt, b: [*]T, ldb: BlasInt) void {
    if (m_ <= 0 or n_ <= 0) return;
    const m = toUsize(m_);
    const n = toUsize(n_);
    if (side == .left) {
        for (0..n) |j| {
            trmv(T, uplo, trans_, diag, m_, a, lda, b + matIndex(ldb, 0, j), 1);
            for (0..m) |i| b[matIndex(ldb, i, j)] = mul(T, alpha, b[matIndex(ldb, i, j)]);
        }
    } else {
        for (0..m) |i| {
            if (opIsUpper(uplo, trans_)) {
                var cc: usize = n;
                while (cc > 0) {
                    cc -= 1;
                    var sum = zero(T);
                    for (0..n) |p| sum = add(T, sum, mul(T, b[matIndex(ldb, i, p)], triValue(T, uplo, diag, trans_, a, lda, p, cc)));
                    b[matIndex(ldb, i, cc)] = mul(T, alpha, sum);
                }
            } else {
                for (0..n) |j| {
                    var sum = zero(T);
                    for (0..n) |p| sum = add(T, sum, mul(T, b[matIndex(ldb, i, p)], triValue(T, uplo, diag, trans_, a, lda, p, j)));
                    b[matIndex(ldb, i, j)] = mul(T, alpha, sum);
                }
            }
        }
    }
}

pub fn trsm(comptime T: type, side: Side, uplo: Uplo, trans_: Order, diag: Diag, m_: BlasInt, n_: BlasInt, alpha: T, a: [*]const T, lda: BlasInt, b: [*]T, ldb: BlasInt) void {
    if (m_ <= 0 or n_ <= 0) return;
    const m = toUsize(m_);
    const n = toUsize(n_);
    if (side == .left) {
        scaleDenseMatrix(T, m, n, alpha, b, ldb);
        for (0..n) |j| trsv(T, uplo, trans_, diag, m_, a, lda, b + matIndex(ldb, 0, j), 1);
    } else {
        scaleDenseMatrix(T, m, n, alpha, b, ldb);
        const right_trans: Order = switch (trans_) {
            .no_trans => .trans,
            .trans => .no_trans,
            .conj_trans => .conj_trans,
        };
        for (0..m) |i| {
            trsv(T, uplo, right_trans, diag, n_, a, lda, b + matIndex(ldb, i, 0), ldb);
        }
    }
}
