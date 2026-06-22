// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Triangular dense, banded, and packed matrix-vector/solve BLAS Level 2 kernels.

const std = @import("std");

const scalar = @import("../scalar.zig");
const indexing = @import("../indexing.zig");
const access = @import("access.zig");

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

const toUsize = indexing.toUsize;
const startIndex = indexing.startIndex;
const matIndex = indexing.matIndex;
const triPackedIndex = indexing.triPackedIndex;
const triBandIndex = indexing.triBandIndex;
const vectorGet = indexing.vectorGet;
const vectorSet = indexing.vectorSet;

const triValue = access.triValue;
const triPackedValue = access.triPackedValue;

pub fn trmv(comptime T: type, uplo: Uplo, trans_: Order, diag: Diag, n_: BlasInt, a: [*]const T, lda: BlasInt, x: [*]T, incx_: BlasInt) void {
    if (n_ <= 0 or incx_ == 0) return;
    const n = toUsize(n_);
    const sx = startIndex(n_, incx_);
    const tmp = std.heap.page_allocator.alloc(T, n) catch return;
    defer std.heap.page_allocator.free(tmp);
    for (0..n) |i| {
        var sum = zero(T);
        for (0..n) |j| sum = add(T, sum, mul(T, triValue(T, uplo, diag, trans_, a, lda, i, j), vectorGet(T, x, sx, j, incx_)));
        tmp[i] = sum;
    }
    for (0..n) |i| vectorSet(T, x, sx, i, incx_, tmp[i]);
}

pub fn tbmv(comptime T: type, uplo: Uplo, trans_: Order, diag: Diag, n_: BlasInt, k_: BlasInt, a: [*]const T, lda: BlasInt, x: [*]T, incx_: BlasInt) void {
    if (n_ <= 0 or incx_ == 0) return;
    const n = toUsize(n_);
    const k = toUsize(k_);
    const sx = startIndex(n_, incx_);
    const tmp = std.heap.page_allocator.alloc(T, n) catch return;
    defer std.heap.page_allocator.free(tmp);
    for (0..n) |i| {
        var sum = zero(T);
        for (0..n) |j| {
            const ar = if (trans_ == .no_trans) i else j;
            const ac = if (trans_ == .no_trans) j else i;
            var av = zero(T);
            if (ar == ac and diag == .unit) {
                av = one(T);
            } else if (triBandIndex(uplo, k, lda, ar, ac)) |idxa| {
                av = a[idxa];
                if (trans_ == .conj_trans) av = conj(T, av);
            }
            sum = add(T, sum, mul(T, av, vectorGet(T, x, sx, j, incx_)));
        }
        tmp[i] = sum;
    }
    for (0..n) |i| vectorSet(T, x, sx, i, incx_, tmp[i]);
}

pub fn tpmv(comptime T: type, uplo: Uplo, trans_: Order, diag: Diag, n_: BlasInt, ap: [*]const T, x: [*]T, incx_: BlasInt) void {
    if (n_ <= 0 or incx_ == 0) return;
    const n = toUsize(n_);
    const sx = startIndex(n_, incx_);
    const tmp = std.heap.page_allocator.alloc(T, n) catch return;
    defer std.heap.page_allocator.free(tmp);
    for (0..n) |i| {
        var sum = zero(T);
        for (0..n) |j| sum = add(T, sum, mul(T, triPackedValue(T, uplo, diag, trans_, n, ap, i, j), vectorGet(T, x, sx, j, incx_)));
        tmp[i] = sum;
    }
    for (0..n) |i| vectorSet(T, x, sx, i, incx_, tmp[i]);
}

pub fn trsv(comptime T: type, uplo: Uplo, trans_: Order, diag: Diag, n_: BlasInt, a: [*]const T, lda: BlasInt, x: [*]T, incx_: BlasInt) void {
    if (n_ <= 0 or incx_ == 0) return;
    const n = toUsize(n_);
    const sx = startIndex(n_, incx_);
    if ((trans_ == .no_trans and uplo == .upper) or (trans_ != .no_trans and uplo == .lower)) {
        var rr: usize = n;
        while (rr > 0) {
            rr -= 1;
            var t = vectorGet(T, x, sx, rr, incx_);
            for (rr + 1..n) |j| t = sub(T, t, mul(T, triValue(T, uplo, diag, trans_, a, lda, rr, j), vectorGet(T, x, sx, j, incx_)));
            if (diag == .non_unit) t = divv(T, t, triValue(T, uplo, diag, trans_, a, lda, rr, rr));
            vectorSet(T, x, sx, rr, incx_, t);
        }
    } else {
        for (0..n) |i| {
            var t = vectorGet(T, x, sx, i, incx_);
            for (0..i) |j| t = sub(T, t, mul(T, triValue(T, uplo, diag, trans_, a, lda, i, j), vectorGet(T, x, sx, j, incx_)));
            if (diag == .non_unit) t = divv(T, t, triValue(T, uplo, diag, trans_, a, lda, i, i));
            vectorSet(T, x, sx, i, incx_, t);
        }
    }
}

pub fn tbsv(comptime T: type, uplo: Uplo, trans_: Order, diag: Diag, n_: BlasInt, k_: BlasInt, a: [*]const T, lda: BlasInt, x: [*]T, incx_: BlasInt) void {
    const n = toUsize(n_);
    const dense = std.heap.page_allocator.alloc(T, n * n) catch return;
    defer std.heap.page_allocator.free(dense);
    @memset(dense, zero(T));
    const k = toUsize(k_);
    for (0..n) |j| for (0..n) |i| {
        if (i == j and diag == .unit) dense[matIndex(@intCast(n), i, j)] = one(T) else if (triBandIndex(uplo, k, lda, i, j)) |idxa| dense[matIndex(@intCast(n), i, j)] = a[idxa];
    };
    trsv(T, uplo, trans_, diag, n_, dense.ptr, @intCast(n), x, incx_);
}

pub fn tpsv(comptime T: type, uplo: Uplo, trans_: Order, diag: Diag, n_: BlasInt, ap: [*]const T, x: [*]T, incx_: BlasInt) void {
    const n = toUsize(n_);
    const dense = std.heap.page_allocator.alloc(T, n * n) catch return;
    defer std.heap.page_allocator.free(dense);
    @memset(dense, zero(T));
    for (0..n) |j| for (0..n) |i| {
        if (i == j and diag == .unit) dense[matIndex(@intCast(n), i, j)] = one(T) else if (triPackedIndex(uplo, n, i, j)) |idxa| dense[matIndex(@intCast(n), i, j)] = ap[idxa];
    };
    trsv(T, uplo, trans_, diag, n_, dense.ptr, @intCast(n), x, incx_);
}
