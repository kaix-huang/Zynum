// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const scalar = @import("scalar.zig");

pub const BlasInt = scalar.BlasInt;
pub const Triangle = scalar.Triangle;
pub const Uplo = scalar.Uplo;

pub fn toUsize(x: BlasInt) usize {
    return @intCast(x);
}

pub fn startIndex(n: BlasInt, inc: BlasInt) isize {
    if (n <= 0) return 0;
    return if (inc >= 0) 0 else @as(isize, 1 - n) * @as(isize, inc);
}

pub fn ix(base: isize, i: usize, inc: BlasInt) usize {
    return @intCast(base + @as(isize, @intCast(i)) * @as(isize, inc));
}

pub fn matIndex(lda: BlasInt, row: usize, col: usize) usize {
    return row + col * toUsize(lda);
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

pub fn bandGeneralIndex(m: usize, n: usize, kl: usize, ku: usize, lda: BlasInt, row: usize, col: usize) ?usize {
    _ = m;
    _ = n;
    if (row + ku < col) return null;
    if (col + kl < row) return null;
    return (ku + row - col) + col * toUsize(lda);
}

pub fn symBandIndex(uplo: Uplo, n: usize, k: usize, lda: BlasInt, row: usize, col: usize) ?usize {
    _ = n;
    if (uplo == .upper) {
        if (row <= col) {
            if (row + k < col) return null;
            return (k + row - col) + col * toUsize(lda);
        }
        if (col + k < row) return null;
        return (k + col - row) + row * toUsize(lda);
    }
    if (row >= col) {
        if (col + k < row) return null;
        return (row - col) + col * toUsize(lda);
    }
    if (row + k < col) return null;
    return (col - row) + row * toUsize(lda);
}

pub fn triBandIndex(uplo: Uplo, k: usize, lda: BlasInt, row: usize, col: usize) ?usize {
    if (uplo == .upper) {
        if (row > col or row + k < col) return null;
        return (k + row - col) + col * toUsize(lda);
    }
    if (row < col or col + k < row) return null;
    return (row - col) + col * toUsize(lda);
}

pub fn vectorGet(comptime T: type, x: [*]const T, start: isize, i: usize, inc: BlasInt) T {
    return x[ix(start, i, inc)];
}

pub fn vectorSet(comptime T: type, x: [*]T, start: isize, i: usize, inc: BlasInt, v: T) void {
    x[ix(start, i, inc)] = v;
}

pub const vectorIndex = ix;
pub const matrixIndex = matIndex;
pub const packedMatrixIndex = packedIndex;
