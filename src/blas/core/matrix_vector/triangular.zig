// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Triangular dense, banded, and packed matrix-vector/solve BLAS Level 2 kernels.

const scalar = @import("../shared/scalar.zig");
const indexing = @import("../shared/indexing.zig");
const vector_ops = @import("../vector/operations.zig");
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
const neg = scalar.neg;
const isComplex = scalar.isComplex;
const isZero = scalar.isZero;

const toUsize = indexing.toUsize;
const startIndex = indexing.startIndex;
const triPackedIndex = indexing.triPackedIndex;
const triBandIndex = indexing.triBandIndex;
const vectorGet = indexing.vectorGet;
const vectorSet = indexing.vectorSet;

const triValue = access.triValue;
const triPackedValue = access.triPackedValue;

fn opIsUpper(uplo: Uplo, trans_: Order) bool {
    return (trans_ == .no_trans and uplo == .upper) or (trans_ != .no_trans and uplo == .lower);
}

const dense_vector_min = 64;
const dense_vector_parallel_min = 512 * 1024;

fn denseAxpy(comptime T: type, n: usize, alpha: T, a: [*]const T, x: [*]T) void {
    if (n == 0 or isZero(T, alpha)) return;
    if (n >= dense_vector_min) {
        if (comptime isComplex(T)) {
            // Keep each triangular dependency step serial. The general complex
            // helper only starts tasking at dense_vector_parallel_min.
            if (n < dense_vector_parallel_min) return vector_ops.axpy(T, @intCast(n), alpha, a, 1, x, 1);
        } else {
            return vector_ops.axpyUnitReal(T, n, alpha, a, x);
        }
    }
    for (0..n) |i| x[i] = add(T, x[i], mul(T, alpha, a[i]));
}

fn denseDot(comptime T: type, n: usize, a: [*]const T, x: [*]const T, conjugate_a: bool) T {
    if (n >= dense_vector_min) {
        if (comptime isComplex(T)) {
            // As above, do not let one solve step fan out into Level 1 tasks.
            if (n < dense_vector_parallel_min) return vector_ops.dot(T, @intCast(n), a, 1, x, 1, conjugate_a);
        } else {
            return vector_ops.dotUnitReal(T, n, a, x);
        }
    }
    var sum = zero(T);
    for (0..n) |i| {
        const av = if (conjugate_a) conj(T, a[i]) else a[i];
        sum = add(T, sum, mul(T, av, x[i]));
    }
    return sum;
}

fn trmvUnit(comptime T: type, uplo: Uplo, trans_: Order, diag: Diag, n: usize, a: [*]const T, lda: BlasInt, x: [*]T) void {
    const lda_usize = toUsize(lda);
    if (trans_ == .no_trans) {
        if (uplo == .upper) {
            for (0..n) |j| {
                const column = a + j * lda_usize;
                const xj = x[j];
                denseAxpy(T, j, xj, column, x);
                if (diag == .non_unit) x[j] = mul(T, xj, column[j]);
            }
        } else {
            var j = n;
            while (j > 0) {
                j -= 1;
                const column = a + j * lda_usize;
                const xj = x[j];
                if (diag == .non_unit) x[j] = mul(T, xj, column[j]);
                denseAxpy(T, n - j - 1, xj, column + j + 1, x + j + 1);
            }
        }
        return;
    }

    const conjugate_a = trans_ == .conj_trans;
    if (uplo == .upper) {
        var j = n;
        while (j > 0) {
            j -= 1;
            const column = a + j * lda_usize;
            var sum = denseDot(T, j, column, x, conjugate_a);
            const diagonal = if (diag == .unit)
                one(T)
            else if (conjugate_a)
                conj(T, column[j])
            else
                column[j];
            sum = add(T, sum, mul(T, diagonal, x[j]));
            x[j] = sum;
        }
    } else {
        for (0..n) |j| {
            const column = a + j * lda_usize;
            const diagonal = if (diag == .unit)
                one(T)
            else if (conjugate_a)
                conj(T, column[j])
            else
                column[j];
            var sum = mul(T, diagonal, x[j]);
            sum = add(T, sum, denseDot(T, n - j - 1, column + j + 1, x + j + 1, conjugate_a));
            x[j] = sum;
        }
    }
}

fn trmvFallback(comptime T: type, uplo: Uplo, trans_: Order, diag: Diag, n_: BlasInt, a: [*]const T, lda: BlasInt, x: [*]T, incx_: BlasInt) void {
    const n = toUsize(n_);
    const sx = startIndex(n_, incx_);
    if (opIsUpper(uplo, trans_)) {
        for (0..n) |i| {
            var sum = zero(T);
            for (0..n) |j| sum = add(T, sum, mul(T, triValue(T, uplo, diag, trans_, a, lda, i, j), vectorGet(T, x, sx, j, incx_)));
            vectorSet(T, x, sx, i, incx_, sum);
        }
    } else {
        var rr: usize = n;
        while (rr > 0) {
            rr -= 1;
            var sum = zero(T);
            for (0..n) |j| sum = add(T, sum, mul(T, triValue(T, uplo, diag, trans_, a, lda, rr, j), vectorGet(T, x, sx, j, incx_)));
            vectorSet(T, x, sx, rr, incx_, sum);
        }
    }
}

fn triBandValue(comptime T: type, uplo: Uplo, trans_: Order, diag: Diag, k: usize, a: [*]const T, lda: BlasInt, row: usize, col: usize) T {
    const ar = if (trans_ == .no_trans) row else col;
    const ac = if (trans_ == .no_trans) col else row;
    if (ar == ac and diag == .unit) return one(T);
    const idxa = triBandIndex(uplo, k, lda, ar, ac) orelse return zero(T);
    const value = a[idxa];
    return if (trans_ == .conj_trans) conj(T, value) else value;
}

noinline fn tbmvBandWindowUnitLeaf(
    comptime T: type,
    comptime uplo: Uplo,
    comptime trans_: Order,
    comptime diag: Diag,
    n: usize,
    k: usize,
    a: [*]const T,
    lda: BlasInt,
    x: [*]T,
) void {
    const lda_usize = toUsize(lda);

    if (comptime trans_ == .no_trans) {
        const matrix_step = lda_usize - 1;
        if (comptime uplo == .upper) {
            for (0..n) |i| {
                var matrix_index = k + i * lda_usize;
                var sum = if (comptime diag == .unit)
                    x[i]
                else
                    mul(T, a[matrix_index], x[i]);

                var j = i + 1;
                const j_end = @min(n, i + k + 1);
                while (j < j_end) : (j += 1) {
                    matrix_index += matrix_step;
                    sum = add(T, sum, mul(T, a[matrix_index], x[j]));
                }
                x[i] = sum;
            }
        } else {
            var i = n;
            while (i > 0) {
                i -= 1;
                const j_start = i - @min(i, k);
                var j = j_start;
                var matrix_index = (i - j_start) + j_start * lda_usize;
                var sum = zero(T);
                while (j < i) : (j += 1) {
                    sum = add(T, sum, mul(T, a[matrix_index], x[j]));
                    matrix_index += matrix_step;
                }
                sum = if (comptime diag == .unit)
                    add(T, sum, x[i])
                else
                    add(T, sum, mul(T, a[matrix_index], x[i]));
                x[i] = sum;
            }
        }
        return;
    }

    if (comptime uplo == .upper) {
        var i = n;
        while (i > 0) {
            i -= 1;
            const j_start = i - @min(i, k);
            var j = j_start;
            var matrix_index = (k - (i - j_start)) + i * lda_usize;
            var sum = zero(T);
            while (j < i) : (j += 1) {
                const av = if (comptime trans_ == .conj_trans) conj(T, a[matrix_index]) else a[matrix_index];
                sum = add(T, sum, mul(T, av, x[j]));
                matrix_index += 1;
            }
            if (comptime diag == .unit) {
                sum = add(T, sum, x[i]);
            } else {
                const av = if (comptime trans_ == .conj_trans) conj(T, a[matrix_index]) else a[matrix_index];
                sum = add(T, sum, mul(T, av, x[i]));
            }
            x[i] = sum;
        }
    } else {
        for (0..n) |i| {
            var matrix_index = i * lda_usize;
            var sum = if (comptime diag == .unit)
                x[i]
            else blk: {
                const av = if (comptime trans_ == .conj_trans) conj(T, a[matrix_index]) else a[matrix_index];
                break :blk mul(T, av, x[i]);
            };

            var j = i + 1;
            const j_end = @min(n, i + k + 1);
            while (j < j_end) : (j += 1) {
                matrix_index += 1;
                const av = if (comptime trans_ == .conj_trans) conj(T, a[matrix_index]) else a[matrix_index];
                sum = add(T, sum, mul(T, av, x[j]));
            }
            x[i] = sum;
        }
    }
}

noinline fn tbmvBandWindowUnitDispatch(comptime T: type, uplo: Uplo, trans_: Order, diag: Diag, n: usize, k: usize, a: [*]const T, lda: BlasInt, x: [*]T) void {
    switch (uplo) {
        .upper => switch (trans_) {
            .no_trans => switch (diag) {
                .unit => tbmvBandWindowUnitLeaf(T, .upper, .no_trans, .unit, n, k, a, lda, x),
                .non_unit => tbmvBandWindowUnitLeaf(T, .upper, .no_trans, .non_unit, n, k, a, lda, x),
            },
            .trans => switch (diag) {
                .unit => tbmvBandWindowUnitLeaf(T, .upper, .trans, .unit, n, k, a, lda, x),
                .non_unit => tbmvBandWindowUnitLeaf(T, .upper, .trans, .non_unit, n, k, a, lda, x),
            },
            .conj_trans => switch (diag) {
                .unit => tbmvBandWindowUnitLeaf(T, .upper, .conj_trans, .unit, n, k, a, lda, x),
                .non_unit => tbmvBandWindowUnitLeaf(T, .upper, .conj_trans, .non_unit, n, k, a, lda, x),
            },
        },
        .lower => switch (trans_) {
            .no_trans => switch (diag) {
                .unit => tbmvBandWindowUnitLeaf(T, .lower, .no_trans, .unit, n, k, a, lda, x),
                .non_unit => tbmvBandWindowUnitLeaf(T, .lower, .no_trans, .non_unit, n, k, a, lda, x),
            },
            .trans => switch (diag) {
                .unit => tbmvBandWindowUnitLeaf(T, .lower, .trans, .unit, n, k, a, lda, x),
                .non_unit => tbmvBandWindowUnitLeaf(T, .lower, .trans, .non_unit, n, k, a, lda, x),
            },
            .conj_trans => switch (diag) {
                .unit => tbmvBandWindowUnitLeaf(T, .lower, .conj_trans, .unit, n, k, a, lda, x),
                .non_unit => tbmvBandWindowUnitLeaf(T, .lower, .conj_trans, .non_unit, n, k, a, lda, x),
            },
        },
    }
}

fn triPackedOpValue(comptime T: type, uplo: Uplo, trans_: Order, diag: Diag, n: usize, ap: [*]const T, row: usize, col: usize) T {
    const ar = if (trans_ == .no_trans) row else col;
    const ac = if (trans_ == .no_trans) col else row;
    if (ar == ac and diag == .unit) return one(T);
    const idxa = triPackedIndex(uplo, n, ar, ac) orelse return zero(T);
    const value = ap[idxa];
    return if (trans_ == .conj_trans) conj(T, value) else value;
}

pub fn trmv(comptime T: type, uplo: Uplo, trans_: Order, diag: Diag, n_: BlasInt, a: [*]const T, lda: BlasInt, x: [*]T, incx_: BlasInt) void {
    if (n_ <= 0 or incx_ == 0) return;
    const n = toUsize(n_);
    if (incx_ == 1) return trmvUnit(T, uplo, trans_, diag, n, a, lda, x);
    trmvFallback(T, uplo, trans_, diag, n_, a, lda, x, incx_);
}

pub fn tbmv(comptime T: type, uplo: Uplo, trans_: Order, diag: Diag, n_: BlasInt, k_: BlasInt, a: [*]const T, lda: BlasInt, x: [*]T, incx_: BlasInt) void {
    if (n_ <= 0 or incx_ == 0) return;
    if (k_ < 0) return;
    const n = toUsize(n_);
    const k = toUsize(k_);
    if (incx_ == 1 and n >= 512 and k <= n / 16) return tbmvBandWindowUnitDispatch(T, uplo, trans_, diag, n, k, a, lda, x);
    const sx = startIndex(n_, incx_);

    if (opIsUpper(uplo, trans_)) {
        for (0..n) |i| {
            var sum = zero(T);
            for (0..n) |j| {
                const av = triBandValue(T, uplo, trans_, diag, k, a, lda, i, j);
                sum = add(T, sum, mul(T, av, vectorGet(T, x, sx, j, incx_)));
            }
            vectorSet(T, x, sx, i, incx_, sum);
        }
    } else {
        var rr: usize = n;
        while (rr > 0) {
            rr -= 1;
            var sum = zero(T);
            for (0..n) |j| {
                const av = triBandValue(T, uplo, trans_, diag, k, a, lda, rr, j);
                sum = add(T, sum, mul(T, av, vectorGet(T, x, sx, j, incx_)));
            }
            vectorSet(T, x, sx, rr, incx_, sum);
        }
    }
}

pub fn tpmv(comptime T: type, uplo: Uplo, trans_: Order, diag: Diag, n_: BlasInt, ap: [*]const T, x: [*]T, incx_: BlasInt) void {
    if (n_ <= 0 or incx_ == 0) return;
    const n = toUsize(n_);
    const sx = startIndex(n_, incx_);
    if (opIsUpper(uplo, trans_)) {
        for (0..n) |i| {
            var sum = zero(T);
            for (0..n) |j| sum = add(T, sum, mul(T, triPackedValue(T, uplo, diag, trans_, n, ap, i, j), vectorGet(T, x, sx, j, incx_)));
            vectorSet(T, x, sx, i, incx_, sum);
        }
    } else {
        var rr: usize = n;
        while (rr > 0) {
            rr -= 1;
            var sum = zero(T);
            for (0..n) |j| sum = add(T, sum, mul(T, triPackedValue(T, uplo, diag, trans_, n, ap, rr, j), vectorGet(T, x, sx, j, incx_)));
            vectorSet(T, x, sx, rr, incx_, sum);
        }
    }
}

fn trsvUnit(comptime T: type, uplo: Uplo, trans_: Order, diag: Diag, n: usize, a: [*]const T, lda: BlasInt, x: [*]T) void {
    const lda_usize = toUsize(lda);
    if (trans_ == .no_trans) {
        if (uplo == .upper) {
            var j = n;
            while (j > 0) {
                j -= 1;
                const column = a + j * lda_usize;
                var xj = x[j];
                if (diag == .non_unit) xj = divv(T, xj, column[j]);
                x[j] = xj;
                denseAxpy(T, j, neg(T, xj), column, x);
            }
        } else {
            for (0..n) |j| {
                const column = a + j * lda_usize;
                var xj = x[j];
                if (diag == .non_unit) xj = divv(T, xj, column[j]);
                x[j] = xj;
                denseAxpy(T, n - j - 1, neg(T, xj), column + j + 1, x + j + 1);
            }
        }
        return;
    }

    const conjugate_a = trans_ == .conj_trans;
    if (uplo == .upper) {
        for (0..n) |j| {
            const column = a + j * lda_usize;
            var xj = sub(T, x[j], denseDot(T, j, column, x, conjugate_a));
            if (diag == .non_unit) {
                const diagonal = if (conjugate_a) conj(T, column[j]) else column[j];
                xj = divv(T, xj, diagonal);
            }
            x[j] = xj;
        }
    } else {
        var j = n;
        while (j > 0) {
            j -= 1;
            const column = a + j * lda_usize;
            var xj = sub(T, x[j], denseDot(T, n - j - 1, column + j + 1, x + j + 1, conjugate_a));
            if (diag == .non_unit) {
                const diagonal = if (conjugate_a) conj(T, column[j]) else column[j];
                xj = divv(T, xj, diagonal);
            }
            x[j] = xj;
        }
    }
}

fn trsvFallback(comptime T: type, uplo: Uplo, trans_: Order, diag: Diag, n_: BlasInt, a: [*]const T, lda: BlasInt, x: [*]T, incx_: BlasInt) void {
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

pub fn trsv(comptime T: type, uplo: Uplo, trans_: Order, diag: Diag, n_: BlasInt, a: [*]const T, lda: BlasInt, x: [*]T, incx_: BlasInt) void {
    if (n_ <= 0 or incx_ == 0) return;
    const n = toUsize(n_);
    if (incx_ == 1) return trsvUnit(T, uplo, trans_, diag, n, a, lda, x);
    trsvFallback(T, uplo, trans_, diag, n_, a, lda, x, incx_);
}

pub fn tbsv(comptime T: type, uplo: Uplo, trans_: Order, diag: Diag, n_: BlasInt, k_: BlasInt, a: [*]const T, lda: BlasInt, x: [*]T, incx_: BlasInt) void {
    if (n_ <= 0 or incx_ == 0) return;
    if (k_ < 0) return;
    const n = toUsize(n_);
    const k = toUsize(k_);
    const sx = startIndex(n_, incx_);
    if ((trans_ == .no_trans and uplo == .upper) or (trans_ != .no_trans and uplo == .lower)) {
        var rr: usize = n;
        while (rr > 0) {
            rr -= 1;
            var t = vectorGet(T, x, sx, rr, incx_);
            for (rr + 1..n) |j| t = sub(T, t, mul(T, triBandValue(T, uplo, trans_, diag, k, a, lda, rr, j), vectorGet(T, x, sx, j, incx_)));
            if (diag == .non_unit) t = divv(T, t, triBandValue(T, uplo, trans_, diag, k, a, lda, rr, rr));
            vectorSet(T, x, sx, rr, incx_, t);
        }
    } else {
        for (0..n) |i| {
            var t = vectorGet(T, x, sx, i, incx_);
            for (0..i) |j| t = sub(T, t, mul(T, triBandValue(T, uplo, trans_, diag, k, a, lda, i, j), vectorGet(T, x, sx, j, incx_)));
            if (diag == .non_unit) t = divv(T, t, triBandValue(T, uplo, trans_, diag, k, a, lda, i, i));
            vectorSet(T, x, sx, i, incx_, t);
        }
    }
}

pub fn tpsv(comptime T: type, uplo: Uplo, trans_: Order, diag: Diag, n_: BlasInt, ap: [*]const T, x: [*]T, incx_: BlasInt) void {
    if (n_ <= 0 or incx_ == 0) return;
    const n = toUsize(n_);
    const sx = startIndex(n_, incx_);
    if ((trans_ == .no_trans and uplo == .upper) or (trans_ != .no_trans and uplo == .lower)) {
        var rr: usize = n;
        while (rr > 0) {
            rr -= 1;
            var t = vectorGet(T, x, sx, rr, incx_);
            for (rr + 1..n) |j| t = sub(T, t, mul(T, triPackedOpValue(T, uplo, trans_, diag, n, ap, rr, j), vectorGet(T, x, sx, j, incx_)));
            if (diag == .non_unit) t = divv(T, t, triPackedOpValue(T, uplo, trans_, diag, n, ap, rr, rr));
            vectorSet(T, x, sx, rr, incx_, t);
        }
    } else {
        for (0..n) |i| {
            var t = vectorGet(T, x, sx, i, incx_);
            for (0..i) |j| t = sub(T, t, mul(T, triPackedOpValue(T, uplo, trans_, diag, n, ap, i, j), vectorGet(T, x, sx, j, incx_)));
            if (diag == .non_unit) t = divv(T, t, triPackedOpValue(T, uplo, trans_, diag, n, ap, i, i));
            vectorSet(T, x, sx, i, incx_, t);
        }
    }
}
