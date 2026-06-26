// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");

const scalar = @import("../scalar.zig");
const indexing = @import("../indexing.zig");
const level2 = @import("../level2.zig");

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
const triValue = level2.triValue;
const trsv = level2.trsv;

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
    const tmp = std.heap.page_allocator.alloc(T, m * n) catch return;
    defer std.heap.page_allocator.free(tmp);
    for (0..n) |j| for (0..m) |i| {
        var sum = zero(T);
        if (side == .left) {
            for (0..m) |p| sum = add(T, sum, mul(T, triValue(T, uplo, diag, trans_, a, lda, i, p), b[matIndex(ldb, p, j)]));
        } else {
            for (0..n) |p| sum = add(T, sum, mul(T, b[matIndex(ldb, i, p)], triValue(T, uplo, diag, trans_, a, lda, p, j)));
        }
        tmp[i + j * m] = mul(T, alpha, sum);
    };
    for (0..n) |j| {
        for (0..m) |i| {
            b[matIndex(ldb, i, j)] = tmp[i + j * m];
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
        const row = std.heap.page_allocator.alloc(T, n) catch return;
        defer std.heap.page_allocator.free(row);
        scaleDenseMatrix(T, m, n, alpha, b, ldb);
        const right_trans: Order = switch (trans_) {
            .no_trans => .trans,
            .trans => .no_trans,
            .conj_trans => .conj_trans,
        };
        for (0..m) |i| {
            for (0..n) |j| row[j] = b[matIndex(ldb, i, j)];
            trsv(T, uplo, right_trans, diag, n_, a, lda, row.ptr, 1);
            for (0..n) |j| b[matIndex(ldb, i, j)] = row[j];
        }
    }
}
