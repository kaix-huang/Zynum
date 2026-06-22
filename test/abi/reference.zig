// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");

pub const Trans = enum { no_trans, trans, conj_trans };
pub const Uplo = enum { upper, lower };
pub const Diag = enum { non_unit, unit };
pub const Side = enum { left, right };

pub const Rng = struct {
    state: u64,

    pub fn init(seed: u64) Rng {
        return .{ .state = if (seed == 0) 0x9e3779b97f4a7c15 else seed };
    }

    fn next(self: *Rng) u64 {
        var x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        return x;
    }

    fn unit(self: *Rng) f64 {
        const bits = self.next() >> 11;
        return @as(f64, @floatFromInt(bits)) / 9007199254740992.0;
    }

    pub fn small(self: *Rng) f64 {
        return (self.unit() * 4.0) - 2.0;
    }

    pub fn scalar(self: *Rng, comptime T: type) T {
        return fromParts(T, self.small(), self.small());
    }
};

pub fn isComplex(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .@"struct" => @hasField(T, "re") and @hasField(T, "im"),
        else => false,
    };
}

pub fn fromParts(comptime T: type, re: f64, im: f64) T {
    if (comptime isComplex(T)) return .{ .re = @floatCast(re), .im = @floatCast(im) };
    return @floatCast(re);
}

pub fn zero(comptime T: type) T {
    return fromParts(T, 0, 0);
}

pub fn one(comptime T: type) T {
    return fromParts(T, 1, 0);
}

pub fn add(comptime T: type, a: T, b: T) T {
    if (comptime isComplex(T)) return .{ .re = a.re + b.re, .im = a.im + b.im };
    return a + b;
}

pub fn sub(comptime T: type, a: T, b: T) T {
    if (comptime isComplex(T)) return .{ .re = a.re - b.re, .im = a.im - b.im };
    return a - b;
}

pub fn mul(comptime T: type, a: T, b: T) T {
    if (comptime isComplex(T)) return .{ .re = a.re * b.re - a.im * b.im, .im = a.re * b.im + a.im * b.re };
    return a * b;
}

pub fn divv(comptime T: type, a: T, b: T) T {
    if (comptime isComplex(T)) {
        const den = b.re * b.re + b.im * b.im;
        return .{ .re = (a.re * b.re + a.im * b.im) / den, .im = (a.im * b.re - a.re * b.im) / den };
    }
    return a / b;
}

pub fn conj(comptime T: type, a: T) T {
    if (comptime isComplex(T)) return .{ .re = a.re, .im = -a.im };
    return a;
}

pub fn maybeConj(comptime T: type, a: T, do_conj: bool) T {
    return if (do_conj) conj(T, a) else a;
}

pub fn expectApprox(comptime T: type, expected: T, actual: T, tol: anytype) !void {
    if (comptime isComplex(T)) {
        try std.testing.expectApproxEqAbs(expected.re, actual.re, tol);
        try std.testing.expectApproxEqAbs(expected.im, actual.im, tol);
    } else {
        try std.testing.expectApproxEqAbs(expected, actual, tol);
    }
}

pub fn vectorStorageLen(n: usize, inc: isize) usize {
    if (n == 0) return 0;
    const step: usize = @intCast(if (inc < 0) -inc else inc);
    return 1 + (n - 1) * step;
}

pub fn vectorIndex(n: usize, inc: isize, i: usize) usize {
    const base: isize = if (inc >= 0) 0 else @as(isize, @intCast(n - 1)) * -inc;
    return @intCast(base + @as(isize, @intCast(i)) * inc);
}

pub fn vectorGet(comptime T: type, x: []const T, n: usize, inc: isize, i: usize) T {
    return x[vectorIndex(n, inc, i)];
}

pub fn vectorSet(comptime T: type, x: []T, n: usize, inc: isize, i: usize, value: T) void {
    x[vectorIndex(n, inc, i)] = value;
}

pub fn fillVector(comptime T: type, rng: *Rng, x: []T, n: usize, inc: isize) void {
    for (0..x.len) |i| x[i] = fromParts(T, 91.0 + @as(f64, @floatFromInt(i)), -41.0);
    for (0..n) |i| vectorSet(T, x, n, inc, i, rng.scalar(T));
}

pub fn colIndex(lda: usize, row: usize, col: usize) usize {
    return row + col * lda;
}

pub fn rowIndex(lda: usize, row: usize, col: usize) usize {
    return row * lda + col;
}

pub fn fillColMajor(comptime T: type, rng: *Rng, a: []T, rows: usize, cols: usize, lda: usize) void {
    for (0..a.len) |i| a[i] = fromParts(T, -73.0 - @as(f64, @floatFromInt(i)), 37.0);
    for (0..cols) |j| {
        for (0..rows) |i| a[colIndex(lda, i, j)] = rng.scalar(T);
    }
}

pub fn fillRowMajor(comptime T: type, rng: *Rng, a: []T, rows: usize, cols: usize, lda: usize) void {
    for (0..a.len) |i| a[i] = fromParts(T, -57.0 - @as(f64, @floatFromInt(i)), 29.0);
    for (0..rows) |i| {
        for (0..cols) |j| a[rowIndex(lda, i, j)] = rng.scalar(T);
    }
}

pub fn colMatrixValue(comptime T: type, trans: Trans, a: []const T, lda: usize, row: usize, col: usize) T {
    return switch (trans) {
        .no_trans => a[colIndex(lda, row, col)],
        .trans => a[colIndex(lda, col, row)],
        .conj_trans => conj(T, a[colIndex(lda, col, row)]),
    };
}

pub fn rowMatrixValue(comptime T: type, trans: Trans, a: []const T, lda: usize, row: usize, col: usize) T {
    return switch (trans) {
        .no_trans => a[rowIndex(lda, row, col)],
        .trans => a[rowIndex(lda, col, row)],
        .conj_trans => conj(T, a[rowIndex(lda, col, row)]),
    };
}

pub fn gemvColMajor(comptime T: type, trans: Trans, m: usize, n: usize, alpha: T, a: []const T, lda: usize, x: []const T, incx: isize, beta: T, y: []T, incy: isize) void {
    const lenx = if (trans == .no_trans) n else m;
    const leny = if (trans == .no_trans) m else n;
    for (0..leny) |i| {
        var sum = zero(T);
        for (0..lenx) |j| sum = add(T, sum, mul(T, colMatrixValue(T, trans, a, lda, i, j), vectorGet(T, x, lenx, incx, j)));
        vectorSet(T, y, leny, incy, i, add(T, mul(T, alpha, sum), mul(T, beta, vectorGet(T, y, leny, incy, i))));
    }
}

pub fn gemvRowMajor(comptime T: type, trans: Trans, m: usize, n: usize, alpha: T, a: []const T, lda: usize, x: []const T, incx: isize, beta: T, y: []T, incy: isize) void {
    const lenx = if (trans == .no_trans) n else m;
    const leny = if (trans == .no_trans) m else n;
    for (0..leny) |i| {
        var sum = zero(T);
        for (0..lenx) |j| sum = add(T, sum, mul(T, rowMatrixValue(T, trans, a, lda, i, j), vectorGet(T, x, lenx, incx, j)));
        vectorSet(T, y, leny, incy, i, add(T, mul(T, alpha, sum), mul(T, beta, vectorGet(T, y, leny, incy, i))));
    }
}

pub fn gemmRowMajor(comptime T: type, transa: Trans, transb: Trans, m: usize, n: usize, k: usize, alpha: T, a: []const T, lda: usize, b: []const T, ldb: usize, beta: T, c: []T, ldc: usize) void {
    for (0..m) |i| {
        for (0..n) |j| {
            var sum = zero(T);
            for (0..k) |p| sum = add(T, sum, mul(T, rowMatrixValue(T, transa, a, lda, i, p), rowMatrixValue(T, transb, b, ldb, p, j)));
            const idx = rowIndex(ldc, i, j);
            c[idx] = add(T, mul(T, alpha, sum), mul(T, beta, c[idx]));
        }
    }
}

pub fn gbIndexColMajor(m: usize, n: usize, kl: usize, ku: usize, lda: usize, row: usize, col: usize) ?usize {
    _ = m;
    _ = n;
    if (row + ku < col) return null;
    if (col + kl < row) return null;
    return (ku + row - col) + col * lda;
}

pub fn gbIndexRowMajor(m: usize, n: usize, kl: usize, ku: usize, lda: usize, row: usize, col: usize) ?usize {
    _ = m;
    _ = n;
    if (row + ku < col) return null;
    if (col + kl < row) return null;
    return row * lda + (kl + col - row);
}

fn gbValueColMajor(comptime T: type, trans: Trans, m: usize, n: usize, kl: usize, ku: usize, a: []const T, lda: usize, row: usize, col: usize) T {
    const ar = if (trans == .no_trans) row else col;
    const ac = if (trans == .no_trans) col else row;
    const idx = gbIndexColMajor(m, n, kl, ku, lda, ar, ac) orelse return zero(T);
    const value = a[idx];
    return if (trans == .conj_trans) conj(T, value) else value;
}

fn gbValueRowMajor(comptime T: type, trans: Trans, m: usize, n: usize, kl: usize, ku: usize, a: []const T, lda: usize, row: usize, col: usize) T {
    const ar = if (trans == .no_trans) row else col;
    const ac = if (trans == .no_trans) col else row;
    const idx = gbIndexRowMajor(m, n, kl, ku, lda, ar, ac) orelse return zero(T);
    const value = a[idx];
    return if (trans == .conj_trans) conj(T, value) else value;
}

pub fn gbmvColMajor(comptime T: type, trans: Trans, m: usize, n: usize, kl: usize, ku: usize, alpha: T, a: []const T, lda: usize, x: []const T, incx: isize, beta: T, y: []T, incy: isize) void {
    const lenx = if (trans == .no_trans) n else m;
    const leny = if (trans == .no_trans) m else n;
    for (0..leny) |i| {
        var sum = zero(T);
        for (0..lenx) |j| sum = add(T, sum, mul(T, gbValueColMajor(T, trans, m, n, kl, ku, a, lda, i, j), vectorGet(T, x, lenx, incx, j)));
        vectorSet(T, y, leny, incy, i, add(T, mul(T, alpha, sum), mul(T, beta, vectorGet(T, y, leny, incy, i))));
    }
}

pub fn gbmvRowMajor(comptime T: type, trans: Trans, m: usize, n: usize, kl: usize, ku: usize, alpha: T, a: []const T, lda: usize, x: []const T, incx: isize, beta: T, y: []T, incy: isize) void {
    const lenx = if (trans == .no_trans) n else m;
    const leny = if (trans == .no_trans) m else n;
    for (0..leny) |i| {
        var sum = zero(T);
        for (0..lenx) |j| sum = add(T, sum, mul(T, gbValueRowMajor(T, trans, m, n, kl, ku, a, lda, i, j), vectorGet(T, x, lenx, incx, j)));
        vectorSet(T, y, leny, incy, i, add(T, mul(T, alpha, sum), mul(T, beta, vectorGet(T, y, leny, incy, i))));
    }
}

pub fn symBandIndex(uplo: Uplo, n: usize, k: usize, lda: usize, row: usize, col: usize) ?usize {
    _ = n;
    if (uplo == .upper) {
        if (row <= col) {
            if (row + k < col) return null;
            return (k + row - col) + col * lda;
        }
        if (col + k < row) return null;
        return (k + col - row) + row * lda;
    }
    if (row >= col) {
        if (col + k < row) return null;
        return (row - col) + col * lda;
    }
    if (row + k < col) return null;
    return (col - row) + row * lda;
}

pub fn packedIndex(uplo: Uplo, n: usize, row: usize, col: usize) usize {
    if (uplo == .upper) {
        if (row <= col) return col * (col + 1) / 2 + row;
        return row * (row + 1) / 2 + col;
    }
    if (row >= col) return col * (2 * n - col + 1) / 2 + (row - col);
    return row * (2 * n - row + 1) / 2 + (col - row);
}

pub fn triPackedIndex(uplo: Uplo, n: usize, row: usize, col: usize) ?usize {
    if (uplo == .upper) {
        if (row > col) return null;
        return col * (col + 1) / 2 + row;
    }
    if (row < col) return null;
    return col * (2 * n - col + 1) / 2 + (row - col);
}

pub fn symPackedValue(comptime T: type, uplo: Uplo, n: usize, ap: []const T, row: usize, col: usize, herm: bool) T {
    var value = ap[packedIndex(uplo, n, row, col)];
    const direct = (uplo == .upper and row <= col) or (uplo == .lower and row >= col);
    if (herm and !direct) value = conj(T, value);
    if (herm and row == col) {
        if (comptime isComplex(T)) value.im = 0;
    }
    return value;
}

pub fn symBandValue(comptime T: type, uplo: Uplo, n: usize, k: usize, a: []const T, lda: usize, row: usize, col: usize, herm: bool) T {
    const idx = symBandIndex(uplo, n, k, lda, row, col) orelse return zero(T);
    var value = a[idx];
    const direct = (uplo == .upper and row <= col) or (uplo == .lower and row >= col);
    if (herm and !direct) value = conj(T, value);
    if (herm and row == col) {
        if (comptime isComplex(T)) value.im = 0;
    }
    return value;
}

pub fn sbmvColMajor(comptime T: type, uplo: Uplo, n: usize, k: usize, alpha: T, a: []const T, lda: usize, x: []const T, incx: isize, beta: T, y: []T, incy: isize, herm: bool) void {
    for (0..n) |i| {
        var sum = zero(T);
        for (0..n) |j| sum = add(T, sum, mul(T, symBandValue(T, uplo, n, k, a, lda, i, j, herm), vectorGet(T, x, n, incx, j)));
        vectorSet(T, y, n, incy, i, add(T, mul(T, alpha, sum), mul(T, beta, vectorGet(T, y, n, incy, i))));
    }
}

pub fn spmvColMajor(comptime T: type, uplo: Uplo, n: usize, alpha: T, ap: []const T, x: []const T, incx: isize, beta: T, y: []T, incy: isize, herm: bool) void {
    for (0..n) |i| {
        var sum = zero(T);
        for (0..n) |j| sum = add(T, sum, mul(T, symPackedValue(T, uplo, n, ap, i, j, herm), vectorGet(T, x, n, incx, j)));
        vectorSet(T, y, n, incy, i, add(T, mul(T, alpha, sum), mul(T, beta, vectorGet(T, y, n, incy, i))));
    }
}

pub fn triValueColMajor(comptime T: type, uplo: Uplo, diag: Diag, trans: Trans, a: []const T, lda: usize, row: usize, col: usize) T {
    const ar = if (trans == .no_trans) row else col;
    const ac = if (trans == .no_trans) col else row;
    if (ar == ac and diag == .unit) return one(T);
    if (uplo == .upper and ar > ac) return zero(T);
    if (uplo == .lower and ar < ac) return zero(T);
    var value = a[colIndex(lda, ar, ac)];
    if (trans == .conj_trans) value = conj(T, value);
    return value;
}

pub fn triValueRowMajor(comptime T: type, uplo: Uplo, diag: Diag, trans: Trans, a: []const T, lda: usize, row: usize, col: usize) T {
    const ar = if (trans == .no_trans) row else col;
    const ac = if (trans == .no_trans) col else row;
    if (ar == ac and diag == .unit) return one(T);
    if (uplo == .upper and ar > ac) return zero(T);
    if (uplo == .lower and ar < ac) return zero(T);
    var value = a[rowIndex(lda, ar, ac)];
    if (trans == .conj_trans) value = conj(T, value);
    return value;
}

pub fn triBandIndex(uplo: Uplo, k: usize, lda: usize, row: usize, col: usize) ?usize {
    if (uplo == .upper) {
        if (row > col or row + k < col) return null;
        return (k + row - col) + col * lda;
    }
    if (row < col or col + k < row) return null;
    return (row - col) + col * lda;
}

pub fn triBandValueColMajor(comptime T: type, uplo: Uplo, diag: Diag, trans: Trans, k: usize, a: []const T, lda: usize, row: usize, col: usize) T {
    const ar = if (trans == .no_trans) row else col;
    const ac = if (trans == .no_trans) col else row;
    if (ar == ac and diag == .unit) return one(T);
    const idx = triBandIndex(uplo, k, lda, ar, ac) orelse return zero(T);
    var value = a[idx];
    if (trans == .conj_trans) value = conj(T, value);
    return value;
}

pub fn triPackedValueColMajor(comptime T: type, uplo: Uplo, diag: Diag, trans: Trans, n: usize, ap: []const T, row: usize, col: usize) T {
    const ar = if (trans == .no_trans) row else col;
    const ac = if (trans == .no_trans) col else row;
    if (ar == ac and diag == .unit) return one(T);
    const idx = triPackedIndex(uplo, n, ar, ac) orelse return zero(T);
    var value = ap[idx];
    if (trans == .conj_trans) value = conj(T, value);
    return value;
}

pub fn trmvColMajor(comptime T: type, uplo: Uplo, trans: Trans, diag: Diag, n: usize, a: []const T, lda: usize, x: []T, incx: isize, work: []T) void {
    for (0..n) |i| {
        var sum = zero(T);
        for (0..n) |j| sum = add(T, sum, mul(T, triValueColMajor(T, uplo, diag, trans, a, lda, i, j), vectorGet(T, x, n, incx, j)));
        work[i] = sum;
    }
    for (0..n) |i| vectorSet(T, x, n, incx, i, work[i]);
}

pub fn trmvRowMajor(comptime T: type, uplo: Uplo, trans: Trans, diag: Diag, n: usize, a: []const T, lda: usize, x: []T, incx: isize, work: []T) void {
    for (0..n) |i| {
        var sum = zero(T);
        for (0..n) |j| sum = add(T, sum, mul(T, triValueRowMajor(T, uplo, diag, trans, a, lda, i, j), vectorGet(T, x, n, incx, j)));
        work[i] = sum;
    }
    for (0..n) |i| vectorSet(T, x, n, incx, i, work[i]);
}

pub fn tbmvColMajor(comptime T: type, uplo: Uplo, trans: Trans, diag: Diag, n: usize, k: usize, a: []const T, lda: usize, x: []T, incx: isize, work: []T) void {
    for (0..n) |i| {
        var sum = zero(T);
        for (0..n) |j| sum = add(T, sum, mul(T, triBandValueColMajor(T, uplo, diag, trans, k, a, lda, i, j), vectorGet(T, x, n, incx, j)));
        work[i] = sum;
    }
    for (0..n) |i| vectorSet(T, x, n, incx, i, work[i]);
}

pub fn tpmvColMajor(comptime T: type, uplo: Uplo, trans: Trans, diag: Diag, n: usize, ap: []const T, x: []T, incx: isize, work: []T) void {
    for (0..n) |i| {
        var sum = zero(T);
        for (0..n) |j| sum = add(T, sum, mul(T, triPackedValueColMajor(T, uplo, diag, trans, n, ap, i, j), vectorGet(T, x, n, incx, j)));
        work[i] = sum;
    }
    for (0..n) |i| vectorSet(T, x, n, incx, i, work[i]);
}

fn opTriUpper(uplo: Uplo, trans: Trans) bool {
    return if (trans == .no_trans) uplo == .upper else uplo == .lower;
}

pub fn trsvColMajor(comptime T: type, uplo: Uplo, trans: Trans, diag: Diag, n: usize, a: []const T, lda: usize, x: []T, incx: isize) void {
    if (opTriUpper(uplo, trans)) {
        var rr = n;
        while (rr > 0) {
            rr -= 1;
            var value = vectorGet(T, x, n, incx, rr);
            for (rr + 1..n) |j| value = sub(T, value, mul(T, triValueColMajor(T, uplo, diag, trans, a, lda, rr, j), vectorGet(T, x, n, incx, j)));
            if (diag == .non_unit) value = divv(T, value, triValueColMajor(T, uplo, diag, trans, a, lda, rr, rr));
            vectorSet(T, x, n, incx, rr, value);
        }
    } else {
        for (0..n) |i| {
            var value = vectorGet(T, x, n, incx, i);
            for (0..i) |j| value = sub(T, value, mul(T, triValueColMajor(T, uplo, diag, trans, a, lda, i, j), vectorGet(T, x, n, incx, j)));
            if (diag == .non_unit) value = divv(T, value, triValueColMajor(T, uplo, diag, trans, a, lda, i, i));
            vectorSet(T, x, n, incx, i, value);
        }
    }
}

pub fn trsvRowMajor(comptime T: type, uplo: Uplo, trans: Trans, diag: Diag, n: usize, a: []const T, lda: usize, x: []T, incx: isize) void {
    if (opTriUpper(uplo, trans)) {
        var rr = n;
        while (rr > 0) {
            rr -= 1;
            var value = vectorGet(T, x, n, incx, rr);
            for (rr + 1..n) |j| value = sub(T, value, mul(T, triValueRowMajor(T, uplo, diag, trans, a, lda, rr, j), vectorGet(T, x, n, incx, j)));
            if (diag == .non_unit) value = divv(T, value, triValueRowMajor(T, uplo, diag, trans, a, lda, rr, rr));
            vectorSet(T, x, n, incx, rr, value);
        }
    } else {
        for (0..n) |i| {
            var value = vectorGet(T, x, n, incx, i);
            for (0..i) |j| value = sub(T, value, mul(T, triValueRowMajor(T, uplo, diag, trans, a, lda, i, j), vectorGet(T, x, n, incx, j)));
            if (diag == .non_unit) value = divv(T, value, triValueRowMajor(T, uplo, diag, trans, a, lda, i, i));
            vectorSet(T, x, n, incx, i, value);
        }
    }
}

pub fn tbsvColMajor(comptime T: type, uplo: Uplo, trans: Trans, diag: Diag, n: usize, k: usize, a: []const T, lda: usize, x: []T, incx: isize) void {
    if (opTriUpper(uplo, trans)) {
        var rr = n;
        while (rr > 0) {
            rr -= 1;
            var value = vectorGet(T, x, n, incx, rr);
            for (rr + 1..n) |j| value = sub(T, value, mul(T, triBandValueColMajor(T, uplo, diag, trans, k, a, lda, rr, j), vectorGet(T, x, n, incx, j)));
            if (diag == .non_unit) value = divv(T, value, triBandValueColMajor(T, uplo, diag, trans, k, a, lda, rr, rr));
            vectorSet(T, x, n, incx, rr, value);
        }
    } else {
        for (0..n) |i| {
            var value = vectorGet(T, x, n, incx, i);
            for (0..i) |j| value = sub(T, value, mul(T, triBandValueColMajor(T, uplo, diag, trans, k, a, lda, i, j), vectorGet(T, x, n, incx, j)));
            if (diag == .non_unit) value = divv(T, value, triBandValueColMajor(T, uplo, diag, trans, k, a, lda, i, i));
            vectorSet(T, x, n, incx, i, value);
        }
    }
}

pub fn tpsvColMajor(comptime T: type, uplo: Uplo, trans: Trans, diag: Diag, n: usize, ap: []const T, x: []T, incx: isize) void {
    if (opTriUpper(uplo, trans)) {
        var rr = n;
        while (rr > 0) {
            rr -= 1;
            var value = vectorGet(T, x, n, incx, rr);
            for (rr + 1..n) |j| value = sub(T, value, mul(T, triPackedValueColMajor(T, uplo, diag, trans, n, ap, rr, j), vectorGet(T, x, n, incx, j)));
            if (diag == .non_unit) value = divv(T, value, triPackedValueColMajor(T, uplo, diag, trans, n, ap, rr, rr));
            vectorSet(T, x, n, incx, rr, value);
        }
    } else {
        for (0..n) |i| {
            var value = vectorGet(T, x, n, incx, i);
            for (0..i) |j| value = sub(T, value, mul(T, triPackedValueColMajor(T, uplo, diag, trans, n, ap, i, j), vectorGet(T, x, n, incx, j)));
            if (diag == .non_unit) value = divv(T, value, triPackedValueColMajor(T, uplo, diag, trans, n, ap, i, i));
            vectorSet(T, x, n, incx, i, value);
        }
    }
}

pub fn trmmRowMajor(comptime T: type, side: Side, uplo: Uplo, trans: Trans, diag: Diag, m: usize, n: usize, alpha: T, a: []const T, lda: usize, b: []T, ldb: usize, work: []T) void {
    for (0..m) |i| {
        for (0..n) |j| {
            var sum = zero(T);
            if (side == .left) {
                for (0..m) |p| sum = add(T, sum, mul(T, triValueRowMajor(T, uplo, diag, trans, a, lda, i, p), b[rowIndex(ldb, p, j)]));
            } else {
                for (0..n) |p| sum = add(T, sum, mul(T, b[rowIndex(ldb, i, p)], triValueRowMajor(T, uplo, diag, trans, a, lda, p, j)));
            }
            work[i * n + j] = mul(T, alpha, sum);
        }
    }
    for (0..m) |i| {
        for (0..n) |j| b[rowIndex(ldb, i, j)] = work[i * n + j];
    }
}

pub fn trsmRowMajor(comptime T: type, side: Side, uplo: Uplo, trans: Trans, diag: Diag, m: usize, n: usize, alpha: T, a: []const T, lda: usize, b: []T, ldb: usize, work: []T) void {
    for (0..m) |i| {
        for (0..n) |j| work[i * n + j] = mul(T, alpha, b[rowIndex(ldb, i, j)]);
    }
    if (side == .left) {
        for (0..n) |j| {
            if (opTriUpper(uplo, trans)) {
                var rr = m;
                while (rr > 0) {
                    rr -= 1;
                    var value = work[rr * n + j];
                    for (rr + 1..m) |p| value = sub(T, value, mul(T, triValueRowMajor(T, uplo, diag, trans, a, lda, rr, p), work[p * n + j]));
                    if (diag == .non_unit) value = divv(T, value, triValueRowMajor(T, uplo, diag, trans, a, lda, rr, rr));
                    work[rr * n + j] = value;
                }
            } else {
                for (0..m) |i| {
                    var value = work[i * n + j];
                    for (0..i) |p| value = sub(T, value, mul(T, triValueRowMajor(T, uplo, diag, trans, a, lda, i, p), work[p * n + j]));
                    if (diag == .non_unit) value = divv(T, value, triValueRowMajor(T, uplo, diag, trans, a, lda, i, i));
                    work[i * n + j] = value;
                }
            }
        }
    } else {
        if (opTriUpper(uplo, trans)) {
            for (0..m) |i| {
                for (0..n) |j| {
                    var value = work[i * n + j];
                    for (0..j) |p| value = sub(T, value, mul(T, work[i * n + p], triValueRowMajor(T, uplo, diag, trans, a, lda, p, j)));
                    if (diag == .non_unit) value = divv(T, value, triValueRowMajor(T, uplo, diag, trans, a, lda, j, j));
                    work[i * n + j] = value;
                }
            }
        } else {
            for (0..m) |i| {
                var cc = n;
                while (cc > 0) {
                    cc -= 1;
                    var value = work[i * n + cc];
                    for (cc + 1..n) |p| value = sub(T, value, mul(T, work[i * n + p], triValueRowMajor(T, uplo, diag, trans, a, lda, p, cc)));
                    if (diag == .non_unit) value = divv(T, value, triValueRowMajor(T, uplo, diag, trans, a, lda, cc, cc));
                    work[i * n + cc] = value;
                }
            }
        }
    }
    for (0..m) |i| {
        for (0..n) |j| b[rowIndex(ldb, i, j)] = work[i * n + j];
    }
}

pub fn gerRowMajor(comptime T: type, m: usize, n: usize, alpha: T, x: []const T, incx: isize, y: []const T, incy: isize, a: []T, lda: usize, conj_y: bool) void {
    for (0..m) |i| {
        for (0..n) |j| {
            const update = mul(T, mul(T, alpha, vectorGet(T, x, m, incx, i)), maybeConj(T, vectorGet(T, y, n, incy, j), conj_y));
            a[rowIndex(lda, i, j)] = add(T, a[rowIndex(lda, i, j)], update);
        }
    }
}

pub fn sprColMajor(comptime T: type, uplo: Uplo, n: usize, alpha: T, x: []const T, incx: isize, ap: []T) void {
    for (0..n) |j| {
        for (0..n) |i| {
            const stored = (uplo == .upper and i <= j) or (uplo == .lower and i >= j);
            if (!stored) continue;
            const idx = packedIndex(uplo, n, i, j);
            ap[idx] = add(T, ap[idx], mul(T, alpha, mul(T, vectorGet(T, x, n, incx, i), vectorGet(T, x, n, incx, j))));
        }
    }
}

pub fn hprColMajor(comptime T: type, uplo: Uplo, n: usize, alpha: anytype, x: []const T, incx: isize, ap: []T) void {
    const alpha_t = fromParts(T, @as(f64, @floatCast(alpha)), 0);
    for (0..n) |j| {
        for (0..n) |i| {
            const stored = (uplo == .upper and i <= j) or (uplo == .lower and i >= j);
            if (!stored) continue;
            const idx = packedIndex(uplo, n, i, j);
            const update = mul(T, alpha_t, mul(T, vectorGet(T, x, n, incx, i), conj(T, vectorGet(T, x, n, incx, j))));
            ap[idx] = add(T, ap[idx], update);
            if (i == j) ap[idx].im = 0;
        }
    }
}

pub fn spr2ColMajor(comptime T: type, uplo: Uplo, n: usize, alpha: T, x: []const T, incx: isize, y: []const T, incy: isize, ap: []T) void {
    for (0..n) |j| {
        for (0..n) |i| {
            const stored = (uplo == .upper and i <= j) or (uplo == .lower and i >= j);
            if (!stored) continue;
            const idx = packedIndex(uplo, n, i, j);
            const update = add(T, mul(T, vectorGet(T, x, n, incx, i), mul(T, alpha, vectorGet(T, y, n, incy, j))), mul(T, vectorGet(T, y, n, incy, i), mul(T, alpha, vectorGet(T, x, n, incx, j))));
            ap[idx] = add(T, ap[idx], update);
        }
    }
}

pub fn hpr2ColMajor(comptime T: type, uplo: Uplo, n: usize, alpha: T, x: []const T, incx: isize, y: []const T, incy: isize, ap: []T) void {
    for (0..n) |j| {
        for (0..n) |i| {
            const stored = (uplo == .upper and i <= j) or (uplo == .lower and i >= j);
            if (!stored) continue;
            const idx = packedIndex(uplo, n, i, j);
            const update = add(
                T,
                mul(T, vectorGet(T, x, n, incx, i), mul(T, alpha, conj(T, vectorGet(T, y, n, incy, j)))),
                mul(T, vectorGet(T, y, n, incy, i), mul(T, conj(T, alpha), conj(T, vectorGet(T, x, n, incx, j)))),
            );
            ap[idx] = add(T, ap[idx], update);
            if (i == j) ap[idx].im = 0;
        }
    }
}
