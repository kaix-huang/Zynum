// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Symmetric, Hermitian, banded, and packed matrix-vector BLAS Level 2 kernels.

const scalar = @import("../scalar.zig");
const indexing = @import("../indexing.zig");
const access = @import("access.zig");

const BlasInt = scalar.BlasInt;
const Uplo = scalar.Uplo;

const zero = scalar.zero;
const add = scalar.add;
const mul = scalar.mul;
const conj = scalar.conj;
const isComplex = scalar.isComplex;
const isZero = scalar.isZero;

const toUsize = indexing.toUsize;
const startIndex = indexing.startIndex;
const ix = indexing.ix;
const symBandIndex = indexing.symBandIndex;
const vectorGet = indexing.vectorGet;

const symValue = access.symValue;
const symPackedValue = access.symPackedValue;

pub fn symv(comptime T: type, uplo: Uplo, n_: BlasInt, alpha: T, a: [*]const T, lda: BlasInt, x: [*]const T, incx_: BlasInt, beta: T, y: [*]T, incy_: BlasInt, herm: bool) void {
    if (n_ <= 0 or incx_ == 0 or incy_ == 0) return;
    const n = toUsize(n_);
    const sx = startIndex(n_, incx_);
    const sy = startIndex(n_, incy_);
    for (0..n) |i| {
        const py = ix(sy, i, incy_);
        y[py] = if (isZero(T, beta)) zero(T) else mul(T, beta, y[py]);
    }
    if (isZero(T, alpha)) return;
    for (0..n) |i| {
        var sum = zero(T);
        for (0..n) |j| sum = add(T, sum, mul(T, symValue(T, uplo, a, lda, i, j, herm), vectorGet(T, x, sx, j, incx_)));
        const py = ix(sy, i, incy_);
        y[py] = add(T, y[py], mul(T, alpha, sum));
    }
}

pub fn sbmv(comptime T: type, uplo: Uplo, n_: BlasInt, k_: BlasInt, alpha: T, a: [*]const T, lda: BlasInt, x: [*]const T, incx_: BlasInt, beta: T, y: [*]T, incy_: BlasInt, herm: bool) void {
    if (n_ <= 0 or incx_ == 0 or incy_ == 0) return;
    const n = toUsize(n_);
    const k = toUsize(k_);
    const sx = startIndex(n_, incx_);
    const sy = startIndex(n_, incy_);
    for (0..n) |i| {
        const py = ix(sy, i, incy_);
        y[py] = if (isZero(T, beta)) zero(T) else mul(T, beta, y[py]);
    }
    if (isZero(T, alpha)) return;
    for (0..n) |i| {
        var sum = zero(T);
        const j0 = if (i > k) i - k else 0;
        const j1 = @min(n, i + k + 1);
        for (j0..j1) |j| {
            if (symBandIndex(uplo, n, k, lda, i, j)) |idxa| {
                var av = a[idxa];
                const direct = (uplo == .upper and i <= j) or (uplo == .lower and i >= j);
                if (herm and !direct) av = conj(T, av);
                if (herm and i == j) {
                    if (comptime isComplex(T)) av.im = 0;
                }
                sum = add(T, sum, mul(T, av, vectorGet(T, x, sx, j, incx_)));
            }
        }
        const py = ix(sy, i, incy_);
        y[py] = add(T, y[py], mul(T, alpha, sum));
    }
}

pub fn spmv(comptime T: type, uplo: Uplo, n_: BlasInt, alpha: T, ap: [*]const T, x: [*]const T, incx_: BlasInt, beta: T, y: [*]T, incy_: BlasInt, herm: bool) void {
    if (n_ <= 0 or incx_ == 0 or incy_ == 0) return;
    const n = toUsize(n_);
    const sx = startIndex(n_, incx_);
    const sy = startIndex(n_, incy_);
    for (0..n) |i| {
        const py = ix(sy, i, incy_);
        y[py] = if (isZero(T, beta)) zero(T) else mul(T, beta, y[py]);
    }
    if (isZero(T, alpha)) return;
    for (0..n) |i| {
        var sum = zero(T);
        for (0..n) |j| sum = add(T, sum, mul(T, symPackedValue(T, uplo, n, ap, i, j, herm), vectorGet(T, x, sx, j, incx_)));
        const py = ix(sy, i, incy_);
        y[py] = add(T, y[py], mul(T, alpha, sum));
    }
}
