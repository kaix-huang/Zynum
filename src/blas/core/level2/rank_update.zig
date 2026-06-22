// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! General, symmetric, Hermitian, and packed rank-update BLAS Level 2 kernels.

const scalar = @import("../scalar.zig");
const indexing = @import("../indexing.zig");

const BlasInt = scalar.BlasInt;
const Uplo = scalar.Uplo;
const Real = scalar.Real;

const realScalar = scalar.realScalar;
const add = scalar.add;
const mul = scalar.mul;
const conj = scalar.conj;
const maybeConj = scalar.maybeConj;
const isZero = scalar.isZero;

const toUsize = indexing.toUsize;
const startIndex = indexing.startIndex;
const matIndex = indexing.matIndex;
const packedIndex = indexing.packedIndex;
const vectorGet = indexing.vectorGet;

pub fn ger(comptime T: type, m_: BlasInt, n_: BlasInt, alpha: T, x: [*]const T, incx_: BlasInt, y: [*]const T, incy_: BlasInt, a: [*]T, lda: BlasInt, conj_y: bool) void {
    if (m_ <= 0 or n_ <= 0 or incx_ == 0 or incy_ == 0 or isZero(T, alpha)) return;
    const m = toUsize(m_);
    const n = toUsize(n_);
    const sx = startIndex(m_, incx_);
    const sy = startIndex(n_, incy_);
    for (0..n) |j| {
        const yj = maybeConj(T, vectorGet(T, y, sy, j, incy_), conj_y);
        const temp = mul(T, alpha, yj);
        for (0..m) |i| {
            const idxa = matIndex(lda, i, j);
            a[idxa] = add(T, a[idxa], mul(T, vectorGet(T, x, sx, i, incx_), temp));
        }
    }
}

pub fn syr(comptime T: type, uplo: Uplo, n_: BlasInt, alpha: T, x: [*]const T, incx_: BlasInt, a: [*]T, lda: BlasInt) void {
    if (n_ <= 0 or incx_ == 0 or isZero(T, alpha)) return;
    const n = toUsize(n_);
    const sx = startIndex(n_, incx_);
    for (0..n) |j| {
        const xj = vectorGet(T, x, sx, j, incx_);
        const temp = mul(T, alpha, xj);
        const row0: usize = if (uplo == .upper) 0 else j;
        const row1: usize = if (uplo == .upper) j + 1 else n;
        for (row0..row1) |i| {
            const idxa = matIndex(lda, i, j);
            a[idxa] = add(T, a[idxa], mul(T, vectorGet(T, x, sx, i, incx_), temp));
        }
    }
}

pub fn spr(comptime T: type, uplo: Uplo, n_: BlasInt, alpha: T, x: [*]const T, incx_: BlasInt, ap: [*]T) void {
    if (n_ <= 0 or incx_ == 0 or isZero(T, alpha)) return;
    const n = toUsize(n_);
    const sx = startIndex(n_, incx_);
    for (0..n) |j| {
        const xj = vectorGet(T, x, sx, j, incx_);
        const temp = mul(T, alpha, xj);
        const row0: usize = if (uplo == .upper) 0 else j;
        const row1: usize = if (uplo == .upper) j + 1 else n;
        for (row0..row1) |i| {
            const idxa = packedIndex(uplo, n, i, j);
            ap[idxa] = add(T, ap[idxa], mul(T, vectorGet(T, x, sx, i, incx_), temp));
        }
    }
}

pub fn syr2(comptime T: type, uplo: Uplo, n_: BlasInt, alpha: T, x: [*]const T, incx_: BlasInt, y: [*]const T, incy_: BlasInt, a: [*]T, lda: BlasInt) void {
    if (n_ <= 0 or incx_ == 0 or incy_ == 0 or isZero(T, alpha)) return;
    const n = toUsize(n_);
    const sx = startIndex(n_, incx_);
    const sy = startIndex(n_, incy_);
    for (0..n) |j| {
        const temp1 = mul(T, alpha, vectorGet(T, y, sy, j, incy_));
        const temp2 = mul(T, alpha, vectorGet(T, x, sx, j, incx_));
        const row0: usize = if (uplo == .upper) 0 else j;
        const row1: usize = if (uplo == .upper) j + 1 else n;
        for (row0..row1) |i| {
            const upd = add(T, mul(T, vectorGet(T, x, sx, i, incx_), temp1), mul(T, vectorGet(T, y, sy, i, incy_), temp2));
            const idxa = matIndex(lda, i, j);
            a[idxa] = add(T, a[idxa], upd);
        }
    }
}

pub fn spr2(comptime T: type, uplo: Uplo, n_: BlasInt, alpha: T, x: [*]const T, incx_: BlasInt, y: [*]const T, incy_: BlasInt, ap: [*]T) void {
    if (n_ <= 0 or incx_ == 0 or incy_ == 0 or isZero(T, alpha)) return;
    const n = toUsize(n_);
    const sx = startIndex(n_, incx_);
    const sy = startIndex(n_, incy_);
    for (0..n) |j| {
        const temp1 = mul(T, alpha, vectorGet(T, y, sy, j, incy_));
        const temp2 = mul(T, alpha, vectorGet(T, x, sx, j, incx_));
        const row0: usize = if (uplo == .upper) 0 else j;
        const row1: usize = if (uplo == .upper) j + 1 else n;
        for (row0..row1) |i| {
            const upd = add(T, mul(T, vectorGet(T, x, sx, i, incx_), temp1), mul(T, vectorGet(T, y, sy, i, incy_), temp2));
            const idxa = packedIndex(uplo, n, i, j);
            ap[idxa] = add(T, ap[idxa], upd);
        }
    }
}

pub fn her(comptime T: type, uplo: Uplo, n_: BlasInt, alpha: Real(T), x: [*]const T, incx_: BlasInt, a: [*]T, lda: BlasInt) void {
    if (n_ <= 0 or incx_ == 0 or alpha == 0) return;
    const n = toUsize(n_);
    const sx = startIndex(n_, incx_);
    for (0..n) |j| {
        const temp = mul(T, realScalar(T, alpha), conj(T, vectorGet(T, x, sx, j, incx_)));
        const row0: usize = if (uplo == .upper) 0 else j;
        const row1: usize = if (uplo == .upper) j + 1 else n;
        for (row0..row1) |i| {
            const idxa = matIndex(lda, i, j);
            a[idxa] = add(T, a[idxa], mul(T, vectorGet(T, x, sx, i, incx_), temp));
            if (i == j) a[idxa].im = 0;
        }
    }
}

pub fn hpr(comptime T: type, uplo: Uplo, n_: BlasInt, alpha: Real(T), x: [*]const T, incx_: BlasInt, ap: [*]T) void {
    if (n_ <= 0 or incx_ == 0 or alpha == 0) return;
    const n = toUsize(n_);
    const sx = startIndex(n_, incx_);
    for (0..n) |j| {
        const temp = mul(T, realScalar(T, alpha), conj(T, vectorGet(T, x, sx, j, incx_)));
        const row0: usize = if (uplo == .upper) 0 else j;
        const row1: usize = if (uplo == .upper) j + 1 else n;
        for (row0..row1) |i| {
            const idxa = packedIndex(uplo, n, i, j);
            ap[idxa] = add(T, ap[idxa], mul(T, vectorGet(T, x, sx, i, incx_), temp));
            if (i == j) ap[idxa].im = 0;
        }
    }
}

pub fn her2(comptime T: type, uplo: Uplo, n_: BlasInt, alpha: T, x: [*]const T, incx_: BlasInt, y: [*]const T, incy_: BlasInt, a: [*]T, lda: BlasInt) void {
    if (n_ <= 0 or incx_ == 0 or incy_ == 0 or isZero(T, alpha)) return;
    const n = toUsize(n_);
    const sx = startIndex(n_, incx_);
    const sy = startIndex(n_, incy_);
    for (0..n) |j| {
        const temp1 = mul(T, alpha, conj(T, vectorGet(T, y, sy, j, incy_)));
        const temp2 = mul(T, conj(T, alpha), conj(T, vectorGet(T, x, sx, j, incx_)));
        const row0: usize = if (uplo == .upper) 0 else j;
        const row1: usize = if (uplo == .upper) j + 1 else n;
        for (row0..row1) |i| {
            const upd = add(T, mul(T, vectorGet(T, x, sx, i, incx_), temp1), mul(T, vectorGet(T, y, sy, i, incy_), temp2));
            const idxa = matIndex(lda, i, j);
            a[idxa] = add(T, a[idxa], upd);
            if (i == j) a[idxa].im = 0;
        }
    }
}

pub fn hpr2(comptime T: type, uplo: Uplo, n_: BlasInt, alpha: T, x: [*]const T, incx_: BlasInt, y: [*]const T, incy_: BlasInt, ap: [*]T) void {
    if (n_ <= 0 or incx_ == 0 or incy_ == 0 or isZero(T, alpha)) return;
    const n = toUsize(n_);
    const sx = startIndex(n_, incx_);
    const sy = startIndex(n_, incy_);
    for (0..n) |j| {
        const temp1 = mul(T, alpha, conj(T, vectorGet(T, y, sy, j, incy_)));
        const temp2 = mul(T, conj(T, alpha), conj(T, vectorGet(T, x, sx, j, incx_)));
        const row0: usize = if (uplo == .upper) 0 else j;
        const row1: usize = if (uplo == .upper) j + 1 else n;
        for (row0..row1) |i| {
            const upd = add(T, mul(T, vectorGet(T, x, sx, i, incx_), temp1), mul(T, vectorGet(T, y, sy, i, incy_), temp2));
            const idxa = packedIndex(uplo, n, i, j);
            ap[idxa] = add(T, ap[idxa], upd);
            if (i == j) ap[idxa].im = 0;
        }
    }
}
