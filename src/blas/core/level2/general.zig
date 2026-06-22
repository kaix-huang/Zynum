// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! General dense and banded matrix-vector BLAS Level 2 kernels.

const scalar = @import("../scalar.zig");
const indexing = @import("../indexing.zig");

const BlasInt = scalar.BlasInt;
const Order = scalar.Order;

const zero = scalar.zero;
const add = scalar.add;
const mul = scalar.mul;
const conj = scalar.conj;
const isZero = scalar.isZero;

const toUsize = indexing.toUsize;
const startIndex = indexing.startIndex;
const ix = indexing.ix;
const matIndex = indexing.matIndex;
const bandGeneralIndex = indexing.bandGeneralIndex;
const vectorGet = indexing.vectorGet;

pub fn gemv(comptime T: type, trans_: Order, m_: BlasInt, n_: BlasInt, alpha: T, a: [*]const T, lda: BlasInt, x: [*]const T, incx_: BlasInt, beta: T, y: [*]T, incy_: BlasInt) void {
    if (m_ <= 0 or n_ <= 0 or incx_ == 0 or incy_ == 0) return;
    const m = toUsize(m_);
    const n = toUsize(n_);
    const lenx: BlasInt = if (trans_ == .no_trans) n_ else m_;
    const leny: BlasInt = if (trans_ == .no_trans) m_ else n_;
    const sx = startIndex(lenx, incx_);
    const sy = startIndex(leny, incy_);
    for (0..toUsize(leny)) |i| {
        const py = ix(sy, i, incy_);
        y[py] = if (isZero(T, beta)) zero(T) else mul(T, beta, y[py]);
    }
    if (isZero(T, alpha)) return;
    if (trans_ == .no_trans) {
        for (0..n) |j| {
            const xj = mul(T, alpha, vectorGet(T, x, sx, j, incx_));
            if (isZero(T, xj)) continue;
            for (0..m) |i| {
                const py = ix(sy, i, incy_);
                y[py] = add(T, y[py], mul(T, a[matIndex(lda, i, j)], xj));
            }
        }
    } else {
        for (0..n) |j| {
            var sum = zero(T);
            for (0..m) |i| {
                var av = a[matIndex(lda, i, j)];
                if (trans_ == .conj_trans) av = conj(T, av);
                sum = add(T, sum, mul(T, av, vectorGet(T, x, sx, i, incx_)));
            }
            const py = ix(sy, j, incy_);
            y[py] = add(T, y[py], mul(T, alpha, sum));
        }
    }
}

pub fn gbmv(comptime T: type, trans_: Order, m_: BlasInt, n_: BlasInt, kl_: BlasInt, ku_: BlasInt, alpha: T, a: [*]const T, lda: BlasInt, x: [*]const T, incx_: BlasInt, beta: T, y: [*]T, incy_: BlasInt) void {
    if (m_ <= 0 or n_ <= 0 or incx_ == 0 or incy_ == 0) return;
    const m = toUsize(m_);
    const n = toUsize(n_);
    const kl = toUsize(kl_);
    const ku = toUsize(ku_);
    const lenx: BlasInt = if (trans_ == .no_trans) n_ else m_;
    const leny: BlasInt = if (trans_ == .no_trans) m_ else n_;
    const sx = startIndex(lenx, incx_);
    const sy = startIndex(leny, incy_);
    for (0..toUsize(leny)) |i| {
        const py = ix(sy, i, incy_);
        y[py] = if (isZero(T, beta)) zero(T) else mul(T, beta, y[py]);
    }
    if (isZero(T, alpha)) return;
    if (trans_ == .no_trans) {
        for (0..n) |j| {
            const xj = mul(T, alpha, vectorGet(T, x, sx, j, incx_));
            const row0 = if (j > ku) j - ku else 0;
            const row1 = @min(m, j + kl + 1);
            for (row0..row1) |i| {
                const idxa = bandGeneralIndex(m, n, kl, ku, lda, i, j).?;
                const py = ix(sy, i, incy_);
                y[py] = add(T, y[py], mul(T, a[idxa], xj));
            }
        }
    } else {
        for (0..n) |j| {
            var sum = zero(T);
            const row0 = if (j > ku) j - ku else 0;
            const row1 = @min(m, j + kl + 1);
            for (row0..row1) |i| {
                var av = a[bandGeneralIndex(m, n, kl, ku, lda, i, j).?];
                if (trans_ == .conj_trans) av = conj(T, av);
                sum = add(T, sum, mul(T, av, vectorGet(T, x, sx, i, incx_)));
            }
            const py = ix(sy, j, incy_);
            y[py] = add(T, y[py], mul(T, alpha, sum));
        }
    }
}
