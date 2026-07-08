// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const types = @import("../types.zig");
const core = @import("../core/unchecked.zig");
const f = @import("fortran.zig");

// CBLAS wrappers keep invalid enum/layout handling local: invalid CBLAS enum
// parameters leave outputs unchanged and return silently. Numeric argument
// validation is delegated to the Fortran ABI wrappers they call.

pub const BlasInt = types.BlasInt;
pub const ComplexF32 = types.ComplexF32;
pub const ComplexF64 = types.ComplexF64;

pub const CblasRowMajor: c_int = 101;
pub const CblasColMajor: c_int = 102;
pub const CblasNoTrans: c_int = 111;
pub const CblasTrans: c_int = 112;
pub const CblasConjTrans: c_int = 113;
pub const CblasUpper: c_int = 121;
pub const CblasLower: c_int = 122;
pub const CblasNonUnit: c_int = 131;
pub const CblasUnit: c_int = 132;
pub const CblasLeft: c_int = 141;
pub const CblasRight: c_int = 142;

fn validLayout(layout: c_int) bool {
    return layout == CblasRowMajor or layout == CblasColMajor;
}

fn validTrans(trans: c_int) bool {
    return trans == CblasNoTrans or trans == CblasTrans or trans == CblasConjTrans;
}

fn validUplo(uplo: c_int) bool {
    return uplo == CblasUpper or uplo == CblasLower;
}

fn validDiag(diag: c_int) bool {
    return diag == CblasNonUnit or diag == CblasUnit;
}

fn validSide(side: c_int) bool {
    return side == CblasLeft or side == CblasRight;
}

fn transChar(trans: c_int) [1]u8 {
    return .{switch (trans) {
        CblasNoTrans => 'N',
        CblasTrans => 'T',
        CblasConjTrans => 'C',
        else => 'X',
    }};
}

fn rowMajorGemvTrans(trans: c_int) [1]u8 {
    return .{switch (trans) {
        CblasNoTrans => 'T',
        CblasTrans, CblasConjTrans => 'N',
        else => 'X',
    }};
}

fn uploChar(uplo: c_int) [1]u8 {
    return .{switch (uplo) {
        CblasUpper => 'U',
        CblasLower => 'L',
        else => 'X',
    }};
}

fn rowMajorUploChar(uplo: c_int) [1]u8 {
    return .{switch (uplo) {
        CblasUpper => 'L',
        CblasLower => 'U',
        else => 'X',
    }};
}

fn sideChar(side: c_int) [1]u8 {
    return .{switch (side) {
        CblasLeft => 'L',
        CblasRight => 'R',
        else => 'X',
    }};
}

fn rowMajorSideChar(side: c_int) [1]u8 {
    return .{switch (side) {
        CblasLeft => 'R',
        CblasRight => 'L',
        else => 'X',
    }};
}

fn diagChar(diag: c_int) [1]u8 {
    return .{switch (diag) {
        CblasNonUnit => 'N',
        CblasUnit => 'U',
        else => 'X',
    }};
}

fn rowMajorTransChar(trans: c_int) [1]u8 {
    return .{switch (trans) {
        CblasNoTrans => 'T',
        CblasTrans => 'N',
        CblasConjTrans => 'C',
        else => 'X',
    }};
}

fn asBlasInt(x: c_int) BlasInt {
    return @intCast(x);
}

fn cblasIndex(i: BlasInt) c_int {
    return if (i <= 0) 0 else @intCast(i - 1);
}

fn max1(x: BlasInt) BlasInt {
    return @max(@as(BlasInt, 1), x);
}

fn rowMajorConjGemv(comptime T: type, m_: BlasInt, n_: BlasInt, alpha: T, a: [*]const T, lda: BlasInt, x: [*]const T, incx_: BlasInt, beta: T, y: [*]T, incy_: BlasInt) void {
    if (m_ <= 0 or n_ <= 0 or lda < max1(n_) or incx_ == 0 or incy_ == 0) return;
    const m: usize = @intCast(m_);
    const n: usize = @intCast(n_);
    const sx = core.startIndex(m_, incx_);
    const sy = core.startIndex(n_, incy_);
    for (0..n) |j| {
        const py = core.ix(sy, j, incy_);
        y[py] = if (core.isZero(T, beta)) core.zero(T) else core.mul(T, beta, y[py]);
    }
    if (core.isZero(T, alpha)) return;
    for (0..n) |j| {
        var sum = core.zero(T);
        for (0..m) |i| {
            const av = core.conj(T, a[i * @as(usize, @intCast(lda)) + j]);
            sum = core.add(T, sum, core.mul(T, av, core.vectorGet(T, x, sx, i, incx_)));
        }
        const py = core.ix(sy, j, incy_);
        y[py] = core.add(T, y[py], core.mul(T, alpha, sum));
    }
}

fn rowMajorConjGbmv(comptime T: type, m_: BlasInt, n_: BlasInt, kl_: BlasInt, ku_: BlasInt, alpha: T, a: [*]const T, lda: BlasInt, x: [*]const T, incx_: BlasInt, beta: T, y: [*]T, incy_: BlasInt) void {
    if (m_ <= 0 or n_ <= 0 or kl_ < 0 or ku_ < 0 or lda < kl_ + ku_ + 1 or incx_ == 0 or incy_ == 0) return;
    const m: usize = @intCast(m_);
    const n: usize = @intCast(n_);
    const kl: usize = @intCast(kl_);
    const ku: usize = @intCast(ku_);
    const ld: usize = @intCast(lda);
    const sx = core.startIndex(m_, incx_);
    const sy = core.startIndex(n_, incy_);
    for (0..n) |j| {
        const py = core.ix(sy, j, incy_);
        y[py] = if (core.isZero(T, beta)) core.zero(T) else core.mul(T, beta, y[py]);
    }
    if (core.isZero(T, alpha)) return;
    for (0..n) |j| {
        var sum = core.zero(T);
        for (0..m) |i| {
            if (i + ku < j) continue;
            if (j + kl < i) continue;
            const band_col: usize = @intCast(@as(isize, @intCast(kl)) + @as(isize, @intCast(j)) - @as(isize, @intCast(i)));
            const av = core.conj(T, a[i * ld + band_col]);
            sum = core.add(T, sum, core.mul(T, av, core.vectorGet(T, x, sx, i, incx_)));
        }
        const py = core.ix(sy, j, incy_);
        y[py] = core.add(T, y[py], core.mul(T, alpha, sum));
    }
}

fn rowMajorGer(comptime T: type, m_: BlasInt, n_: BlasInt, alpha: T, x: [*]const T, incx_: BlasInt, y: [*]const T, incy_: BlasInt, a: [*]T, lda: BlasInt, conj_y: bool) void {
    if (m_ <= 0 or n_ <= 0 or lda < max1(n_) or incx_ == 0 or incy_ == 0 or core.isZero(T, alpha)) return;
    const m: usize = @intCast(m_);
    const n: usize = @intCast(n_);
    const sx = core.startIndex(m_, incx_);
    const sy = core.startIndex(n_, incy_);
    const ld: usize = @intCast(lda);
    for (0..m) |row| {
        const x_value = core.vectorGet(T, x, sx, row, incx_);
        if (core.isZero(T, x_value)) continue;
        const scaled_x = core.mul(T, alpha, x_value);
        for (0..n) |col| {
            const y_value = core.maybeConj(T, core.vectorGet(T, y, sy, col, incy_), conj_y);
            const index = row * ld + col;
            a[index] = core.add(T, a[index], core.mul(T, scaled_x, y_value));
        }
    }
}

fn rowMajorConjTriValue(comptime T: type, uplo: c_int, diag: c_int, a: [*]const T, lda: BlasInt, row: usize, col: usize) T {
    if (row == col and diag == CblasUnit) return core.one(T);
    const a_row = col;
    const a_col = row;
    if (uplo == CblasUpper and a_row > a_col) return core.zero(T);
    if (uplo == CblasLower and a_row < a_col) return core.zero(T);
    const ld: usize = @intCast(lda);
    return core.conj(T, a[a_row * ld + a_col]);
}

fn rowMajorTriBandIndex(uplo: c_int, k: usize, lda: BlasInt, row: usize, col: usize) ?usize {
    const ld: usize = @intCast(lda);
    if (uplo == CblasUpper) {
        if (row > col or row + k < col) return null;
        return row * ld + (col - row);
    }
    if (row < col or col + k < row) return null;
    return row * ld + (k + col - row);
}

fn rowMajorTriPackedIndex(uplo: c_int, n: usize, row: usize, col: usize) ?usize {
    if (uplo == CblasUpper) {
        if (row > col) return null;
        return row * n - row * (row + 1) / 2 + col;
    }
    if (row < col) return null;
    return row * (row + 1) / 2 + col;
}

fn rowMajorConjTriBandValue(comptime T: type, uplo: c_int, diag: c_int, k: usize, a: [*]const T, lda: BlasInt, row: usize, col: usize) T {
    if (row == col and diag == CblasUnit) return core.one(T);
    const a_row = col;
    const a_col = row;
    const idx = rowMajorTriBandIndex(uplo, k, lda, a_row, a_col) orelse return core.zero(T);
    return core.conj(T, a[idx]);
}

fn rowMajorConjTriPackedValue(comptime T: type, uplo: c_int, diag: c_int, n: usize, ap: [*]const T, row: usize, col: usize) T {
    if (row == col and diag == CblasUnit) return core.one(T);
    const a_row = col;
    const a_col = row;
    const idx = rowMajorTriPackedIndex(uplo, n, a_row, a_col) orelse return core.zero(T);
    return core.conj(T, ap[idx]);
}

fn rowMajorConjTransOpUpper(uplo: c_int) bool {
    return uplo == CblasLower;
}

fn rowMajorConjTrmv(comptime T: type, uplo: c_int, diag: c_int, n_: BlasInt, a: [*]const T, lda: BlasInt, x: [*]T, incx_: BlasInt) void {
    if (!validUplo(uplo) or !validDiag(diag) or n_ <= 0 or lda < max1(n_) or incx_ == 0) return;
    const n: usize = @intCast(n_);
    const sx = core.startIndex(n_, incx_);
    if (rowMajorConjTransOpUpper(uplo)) {
        for (0..n) |i| {
            var sum = core.zero(T);
            for (i..n) |j| {
                sum = core.add(T, sum, core.mul(T, rowMajorConjTriValue(T, uplo, diag, a, lda, i, j), core.vectorGet(T, x, sx, j, incx_)));
            }
            core.vectorSet(T, x, sx, i, incx_, sum);
        }
    } else {
        var rr = n;
        while (rr > 0) {
            rr -= 1;
            var sum = core.zero(T);
            for (0..rr + 1) |j| {
                sum = core.add(T, sum, core.mul(T, rowMajorConjTriValue(T, uplo, diag, a, lda, rr, j), core.vectorGet(T, x, sx, j, incx_)));
            }
            core.vectorSet(T, x, sx, rr, incx_, sum);
        }
    }
}

fn rowMajorConjTrsv(comptime T: type, uplo: c_int, diag: c_int, n_: BlasInt, a: [*]const T, lda: BlasInt, x: [*]T, incx_: BlasInt) void {
    if (!validUplo(uplo) or !validDiag(diag) or n_ <= 0 or lda < max1(n_) or incx_ == 0) return;
    const n: usize = @intCast(n_);
    const sx = core.startIndex(n_, incx_);
    if (rowMajorConjTransOpUpper(uplo)) {
        var rr = n;
        while (rr > 0) {
            rr -= 1;
            var value = core.vectorGet(T, x, sx, rr, incx_);
            for (rr + 1..n) |j| {
                value = core.sub(T, value, core.mul(T, rowMajorConjTriValue(T, uplo, diag, a, lda, rr, j), core.vectorGet(T, x, sx, j, incx_)));
            }
            if (diag != CblasUnit) value = core.divv(T, value, rowMajorConjTriValue(T, uplo, diag, a, lda, rr, rr));
            core.vectorSet(T, x, sx, rr, incx_, value);
        }
    } else {
        for (0..n) |i| {
            var value = core.vectorGet(T, x, sx, i, incx_);
            for (0..i) |j| {
                value = core.sub(T, value, core.mul(T, rowMajorConjTriValue(T, uplo, diag, a, lda, i, j), core.vectorGet(T, x, sx, j, incx_)));
            }
            if (diag != CblasUnit) value = core.divv(T, value, rowMajorConjTriValue(T, uplo, diag, a, lda, i, i));
            core.vectorSet(T, x, sx, i, incx_, value);
        }
    }
}

fn rowMajorConjTbmv(comptime T: type, uplo: c_int, diag: c_int, n_: BlasInt, k_: BlasInt, a: [*]const T, lda: BlasInt, x: [*]T, incx_: BlasInt) void {
    if (!validUplo(uplo) or !validDiag(diag) or n_ <= 0 or k_ < 0 or lda < k_ + 1 or incx_ == 0) return;
    const n: usize = @intCast(n_);
    const k: usize = @intCast(k_);
    const sx = core.startIndex(n_, incx_);
    if (rowMajorConjTransOpUpper(uplo)) {
        for (0..n) |i| {
            var sum = core.zero(T);
            for (i..n) |j| {
                sum = core.add(T, sum, core.mul(T, rowMajorConjTriBandValue(T, uplo, diag, k, a, lda, i, j), core.vectorGet(T, x, sx, j, incx_)));
            }
            core.vectorSet(T, x, sx, i, incx_, sum);
        }
    } else {
        var rr = n;
        while (rr > 0) {
            rr -= 1;
            var sum = core.zero(T);
            for (0..rr + 1) |j| {
                sum = core.add(T, sum, core.mul(T, rowMajorConjTriBandValue(T, uplo, diag, k, a, lda, rr, j), core.vectorGet(T, x, sx, j, incx_)));
            }
            core.vectorSet(T, x, sx, rr, incx_, sum);
        }
    }
}

fn rowMajorConjTpmv(comptime T: type, uplo: c_int, diag: c_int, n_: BlasInt, ap: [*]const T, x: [*]T, incx_: BlasInt) void {
    if (!validUplo(uplo) or !validDiag(diag) or n_ <= 0 or incx_ == 0) return;
    const n: usize = @intCast(n_);
    const sx = core.startIndex(n_, incx_);
    if (rowMajorConjTransOpUpper(uplo)) {
        for (0..n) |i| {
            var sum = core.zero(T);
            for (i..n) |j| {
                sum = core.add(T, sum, core.mul(T, rowMajorConjTriPackedValue(T, uplo, diag, n, ap, i, j), core.vectorGet(T, x, sx, j, incx_)));
            }
            core.vectorSet(T, x, sx, i, incx_, sum);
        }
    } else {
        var rr = n;
        while (rr > 0) {
            rr -= 1;
            var sum = core.zero(T);
            for (0..rr + 1) |j| {
                sum = core.add(T, sum, core.mul(T, rowMajorConjTriPackedValue(T, uplo, diag, n, ap, rr, j), core.vectorGet(T, x, sx, j, incx_)));
            }
            core.vectorSet(T, x, sx, rr, incx_, sum);
        }
    }
}

fn rowMajorConjTbsv(comptime T: type, uplo: c_int, diag: c_int, n_: BlasInt, k_: BlasInt, a: [*]const T, lda: BlasInt, x: [*]T, incx_: BlasInt) void {
    if (!validUplo(uplo) or !validDiag(diag) or n_ <= 0 or k_ < 0 or lda < k_ + 1 or incx_ == 0) return;
    const n: usize = @intCast(n_);
    const k: usize = @intCast(k_);
    const sx = core.startIndex(n_, incx_);
    if (rowMajorConjTransOpUpper(uplo)) {
        var rr = n;
        while (rr > 0) {
            rr -= 1;
            var value = core.vectorGet(T, x, sx, rr, incx_);
            for (rr + 1..n) |j| {
                value = core.sub(T, value, core.mul(T, rowMajorConjTriBandValue(T, uplo, diag, k, a, lda, rr, j), core.vectorGet(T, x, sx, j, incx_)));
            }
            if (diag != CblasUnit) value = core.divv(T, value, rowMajorConjTriBandValue(T, uplo, diag, k, a, lda, rr, rr));
            core.vectorSet(T, x, sx, rr, incx_, value);
        }
    } else {
        for (0..n) |i| {
            var value = core.vectorGet(T, x, sx, i, incx_);
            for (0..i) |j| {
                value = core.sub(T, value, core.mul(T, rowMajorConjTriBandValue(T, uplo, diag, k, a, lda, i, j), core.vectorGet(T, x, sx, j, incx_)));
            }
            if (diag != CblasUnit) value = core.divv(T, value, rowMajorConjTriBandValue(T, uplo, diag, k, a, lda, i, i));
            core.vectorSet(T, x, sx, i, incx_, value);
        }
    }
}

fn rowMajorConjTpsv(comptime T: type, uplo: c_int, diag: c_int, n_: BlasInt, ap: [*]const T, x: [*]T, incx_: BlasInt) void {
    if (!validUplo(uplo) or !validDiag(diag) or n_ <= 0 or incx_ == 0) return;
    const n: usize = @intCast(n_);
    const sx = core.startIndex(n_, incx_);
    if (rowMajorConjTransOpUpper(uplo)) {
        var rr = n;
        while (rr > 0) {
            rr -= 1;
            var value = core.vectorGet(T, x, sx, rr, incx_);
            for (rr + 1..n) |j| {
                value = core.sub(T, value, core.mul(T, rowMajorConjTriPackedValue(T, uplo, diag, n, ap, rr, j), core.vectorGet(T, x, sx, j, incx_)));
            }
            if (diag != CblasUnit) value = core.divv(T, value, rowMajorConjTriPackedValue(T, uplo, diag, n, ap, rr, rr));
            core.vectorSet(T, x, sx, rr, incx_, value);
        }
    } else {
        for (0..n) |i| {
            var value = core.vectorGet(T, x, sx, i, incx_);
            for (0..i) |j| {
                value = core.sub(T, value, core.mul(T, rowMajorConjTriPackedValue(T, uplo, diag, n, ap, i, j), core.vectorGet(T, x, sx, j, incx_)));
            }
            if (diag != CblasUnit) value = core.divv(T, value, rowMajorConjTriPackedValue(T, uplo, diag, n, ap, i, i));
            core.vectorSet(T, x, sx, i, incx_, value);
        }
    }
}

// Level 1 CBLAS wrappers.
pub export fn cblas_sswap(n: c_int, x: [*]f32, incx: c_int, y: [*]f32, incy: c_int) callconv(.c) void {
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.sswap_(&nn, x, &ix, y, &iy);
}
pub export fn cblas_dswap(n: c_int, x: [*]f64, incx: c_int, y: [*]f64, incy: c_int) callconv(.c) void {
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.dswap_(&nn, x, &ix, y, &iy);
}
pub export fn cblas_cswap(n: c_int, x: [*]ComplexF32, incx: c_int, y: [*]ComplexF32, incy: c_int) callconv(.c) void {
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.cswap_(&nn, x, &ix, y, &iy);
}
pub export fn cblas_zswap(n: c_int, x: [*]ComplexF64, incx: c_int, y: [*]ComplexF64, incy: c_int) callconv(.c) void {
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.zswap_(&nn, x, &ix, y, &iy);
}

pub export fn cblas_scopy(n: c_int, x: [*]const f32, incx: c_int, y: [*]f32, incy: c_int) callconv(.c) void {
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.scopy_(&nn, x, &ix, y, &iy);
}
pub export fn cblas_dcopy(n: c_int, x: [*]const f64, incx: c_int, y: [*]f64, incy: c_int) callconv(.c) void {
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.dcopy_(&nn, x, &ix, y, &iy);
}
pub export fn cblas_ccopy(n: c_int, x: [*]const ComplexF32, incx: c_int, y: [*]ComplexF32, incy: c_int) callconv(.c) void {
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.ccopy_(&nn, x, &ix, y, &iy);
}
pub export fn cblas_zcopy(n: c_int, x: [*]const ComplexF64, incx: c_int, y: [*]ComplexF64, incy: c_int) callconv(.c) void {
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.zcopy_(&nn, x, &ix, y, &iy);
}

pub export fn cblas_saxpy(n: c_int, alpha: f32, x: [*]const f32, incx: c_int, y: [*]f32, incy: c_int) callconv(.c) void {
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.saxpy_(&nn, &alpha, x, &ix, y, &iy);
}
pub export fn cblas_daxpy(n: c_int, alpha: f64, x: [*]const f64, incx: c_int, y: [*]f64, incy: c_int) callconv(.c) void {
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.daxpy_(&nn, &alpha, x, &ix, y, &iy);
}
pub export fn cblas_caxpy(n: c_int, alpha: *const ComplexF32, x: [*]const ComplexF32, incx: c_int, y: [*]ComplexF32, incy: c_int) callconv(.c) void {
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.caxpy_(&nn, alpha, x, &ix, y, &iy);
}
pub export fn cblas_zaxpy(n: c_int, alpha: *const ComplexF64, x: [*]const ComplexF64, incx: c_int, y: [*]ComplexF64, incy: c_int) callconv(.c) void {
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.zaxpy_(&nn, alpha, x, &ix, y, &iy);
}

pub export fn cblas_saxpby(n: c_int, alpha: f32, x: [*]const f32, incx: c_int, beta: f32, y: [*]f32, incy: c_int) callconv(.c) void {
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.saxpby_(&nn, &alpha, x, &ix, &beta, y, &iy);
}
pub export fn cblas_daxpby(n: c_int, alpha: f64, x: [*]const f64, incx: c_int, beta: f64, y: [*]f64, incy: c_int) callconv(.c) void {
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.daxpby_(&nn, &alpha, x, &ix, &beta, y, &iy);
}
pub export fn cblas_caxpby(n: c_int, alpha: *const ComplexF32, x: [*]const ComplexF32, incx: c_int, beta: *const ComplexF32, y: [*]ComplexF32, incy: c_int) callconv(.c) void {
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.caxpby_(&nn, alpha, x, &ix, beta, y, &iy);
}
pub export fn cblas_zaxpby(n: c_int, alpha: *const ComplexF64, x: [*]const ComplexF64, incx: c_int, beta: *const ComplexF64, y: [*]ComplexF64, incy: c_int) callconv(.c) void {
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.zaxpby_(&nn, alpha, x, &ix, beta, y, &iy);
}

pub export fn cblas_sscal(n: c_int, alpha: f32, x: [*]f32, incx: c_int) callconv(.c) void {
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    f.sscal_(&nn, &alpha, x, &ix);
}
pub export fn cblas_dscal(n: c_int, alpha: f64, x: [*]f64, incx: c_int) callconv(.c) void {
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    f.dscal_(&nn, &alpha, x, &ix);
}
pub export fn cblas_cscal(n: c_int, alpha: *const ComplexF32, x: [*]ComplexF32, incx: c_int) callconv(.c) void {
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    f.cscal_(&nn, alpha, x, &ix);
}
pub export fn cblas_zscal(n: c_int, alpha: *const ComplexF64, x: [*]ComplexF64, incx: c_int) callconv(.c) void {
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    f.zscal_(&nn, alpha, x, &ix);
}
pub export fn cblas_csscal(n: c_int, alpha: f32, x: [*]ComplexF32, incx: c_int) callconv(.c) void {
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    f.csscal_(&nn, &alpha, x, &ix);
}
pub export fn cblas_zdscal(n: c_int, alpha: f64, x: [*]ComplexF64, incx: c_int) callconv(.c) void {
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    f.zdscal_(&nn, &alpha, x, &ix);
}

pub export fn cblas_sdot(n: c_int, x: [*]const f32, incx: c_int, y: [*]const f32, incy: c_int) callconv(.c) f32 {
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    return f.sdot_(&nn, x, &ix, y, &iy);
}
pub export fn cblas_ddot(n: c_int, x: [*]const f64, incx: c_int, y: [*]const f64, incy: c_int) callconv(.c) f64 {
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    return f.ddot_(&nn, x, &ix, y, &iy);
}
pub export fn cblas_cdotu_sub(n: c_int, x: [*]const ComplexF32, incx: c_int, y: [*]const ComplexF32, incy: c_int, out: *ComplexF32) callconv(.c) void {
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.cdotu_sub_(&nn, x, &ix, y, &iy, out);
}
pub export fn cblas_zdotu_sub(n: c_int, x: [*]const ComplexF64, incx: c_int, y: [*]const ComplexF64, incy: c_int, out: *ComplexF64) callconv(.c) void {
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.zdotu_sub_(&nn, x, &ix, y, &iy, out);
}
pub export fn cblas_cdotc_sub(n: c_int, x: [*]const ComplexF32, incx: c_int, y: [*]const ComplexF32, incy: c_int, out: *ComplexF32) callconv(.c) void {
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.cdotc_sub_(&nn, x, &ix, y, &iy, out);
}
pub export fn cblas_zdotc_sub(n: c_int, x: [*]const ComplexF64, incx: c_int, y: [*]const ComplexF64, incy: c_int, out: *ComplexF64) callconv(.c) void {
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.zdotc_sub_(&nn, x, &ix, y, &iy, out);
}

pub export fn cblas_snrm2(n: c_int, x: [*]const f32, incx: c_int) callconv(.c) f32 {
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    return f.snrm2_(&nn, x, &ix);
}
pub export fn cblas_dnrm2(n: c_int, x: [*]const f64, incx: c_int) callconv(.c) f64 {
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    return f.dnrm2_(&nn, x, &ix);
}
pub export fn cblas_scnrm2(n: c_int, x: [*]const ComplexF32, incx: c_int) callconv(.c) f32 {
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    return f.scnrm2_(&nn, x, &ix);
}
pub export fn cblas_dznrm2(n: c_int, x: [*]const ComplexF64, incx: c_int) callconv(.c) f64 {
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    return f.dznrm2_(&nn, x, &ix);
}

pub export fn cblas_sasum(n: c_int, x: [*]const f32, incx: c_int) callconv(.c) f32 {
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    return f.sasum_(&nn, x, &ix);
}
pub export fn cblas_dasum(n: c_int, x: [*]const f64, incx: c_int) callconv(.c) f64 {
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    return f.dasum_(&nn, x, &ix);
}
pub export fn cblas_scasum(n: c_int, x: [*]const ComplexF32, incx: c_int) callconv(.c) f32 {
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    return f.scasum_(&nn, x, &ix);
}
pub export fn cblas_dzasum(n: c_int, x: [*]const ComplexF64, incx: c_int) callconv(.c) f64 {
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    return f.dzasum_(&nn, x, &ix);
}

pub export fn cblas_isamax(n: c_int, x: [*]const f32, incx: c_int) callconv(.c) c_int {
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    return cblasIndex(f.isamax_(&nn, x, &ix));
}
pub export fn cblas_idamax(n: c_int, x: [*]const f64, incx: c_int) callconv(.c) c_int {
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    return cblasIndex(f.idamax_(&nn, x, &ix));
}
pub export fn cblas_icamax(n: c_int, x: [*]const ComplexF32, incx: c_int) callconv(.c) c_int {
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    return cblasIndex(f.icamax_(&nn, x, &ix));
}
pub export fn cblas_izamax(n: c_int, x: [*]const ComplexF64, incx: c_int) callconv(.c) c_int {
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    return cblasIndex(f.izamax_(&nn, x, &ix));
}

pub export fn cblas_sdsdot(n: c_int, sb: f32, x: [*]const f32, incx: c_int, y: [*]const f32, incy: c_int) callconv(.c) f32 {
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    return f.sdsdot_(&nn, &sb, x, &ix, y, &iy);
}
pub export fn cblas_dsdot(n: c_int, x: [*]const f32, incx: c_int, y: [*]const f32, incy: c_int) callconv(.c) f64 {
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    return f.dsdot_(&nn, x, &ix, y, &iy);
}

pub export fn cblas_srotg(a: *f32, b: *f32, c: *f32, s: *f32) callconv(.c) void {
    f.srotg_(a, b, c, s);
}
pub export fn cblas_drotg(a: *f64, b: *f64, c: *f64, s: *f64) callconv(.c) void {
    f.drotg_(a, b, c, s);
}
pub export fn cblas_crotg(a: *ComplexF32, b: *ComplexF32, c: *f32, s: *ComplexF32) callconv(.c) void {
    f.crotg_(a, b, c, s);
}
pub export fn cblas_zrotg(a: *ComplexF64, b: *ComplexF64, c: *f64, s: *ComplexF64) callconv(.c) void {
    f.zrotg_(a, b, c, s);
}

pub export fn cblas_srot(n: c_int, x: [*]f32, incx: c_int, y: [*]f32, incy: c_int, c: f32, s: f32) callconv(.c) void {
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.srot_(&nn, x, &ix, y, &iy, &c, &s);
}
pub export fn cblas_drot(n: c_int, x: [*]f64, incx: c_int, y: [*]f64, incy: c_int, c: f64, s: f64) callconv(.c) void {
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.drot_(&nn, x, &ix, y, &iy, &c, &s);
}
pub export fn cblas_csrot(n: c_int, x: [*]ComplexF32, incx: c_int, y: [*]ComplexF32, incy: c_int, c: f32, s: f32) callconv(.c) void {
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.csrot_(&nn, x, &ix, y, &iy, &c, &s);
}
pub export fn cblas_zdrot(n: c_int, x: [*]ComplexF64, incx: c_int, y: [*]ComplexF64, incy: c_int, c: f64, s: f64) callconv(.c) void {
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.zdrot_(&nn, x, &ix, y, &iy, &c, &s);
}

pub export fn cblas_srotm(n: c_int, x: [*]f32, incx: c_int, y: [*]f32, incy: c_int, param: [*]const f32) callconv(.c) void {
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.srotm_(&nn, x, &ix, y, &iy, param);
}
pub export fn cblas_drotm(n: c_int, x: [*]f64, incx: c_int, y: [*]f64, incy: c_int, param: [*]const f64) callconv(.c) void {
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.drotm_(&nn, x, &ix, y, &iy, param);
}
pub export fn cblas_srotmg(d1: *f32, d2: *f32, x1: *f32, y1: f32, param: [*]f32) callconv(.c) void {
    f.srotmg_(d1, d2, x1, &y1, param);
}
pub export fn cblas_drotmg(d1: *f64, d2: *f64, x1: *f64, y1: f64, param: [*]f64) callconv(.c) void {
    f.drotmg_(d1, d2, x1, &y1, param);
}

// Level 2 CBLAS wrappers.
pub export fn cblas_sgemv(layout: c_int, trans: c_int, m: c_int, n: c_int, alpha: f32, a: [*]const f32, lda: c_int, x: [*]const f32, incx: c_int, beta: f32, y: [*]f32, incy: c_int) callconv(.c) void {
    if (!validLayout(layout)) return;
    var mm = asBlasInt(if (layout == CblasRowMajor) n else m);
    var nn = asBlasInt(if (layout == CblasRowMajor) m else n);
    var tt = if (layout == CblasRowMajor) rowMajorGemvTrans(trans) else transChar(trans);
    var aa = asBlasInt(lda);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.sgemv_(&tt, &mm, &nn, &alpha, a, &aa, x, &ix, &beta, y, &iy);
}

pub export fn cblas_dgemv(layout: c_int, trans: c_int, m: c_int, n: c_int, alpha: f64, a: [*]const f64, lda: c_int, x: [*]const f64, incx: c_int, beta: f64, y: [*]f64, incy: c_int) callconv(.c) void {
    if (!validLayout(layout)) return;
    var mm = asBlasInt(if (layout == CblasRowMajor) n else m);
    var nn = asBlasInt(if (layout == CblasRowMajor) m else n);
    var tt = if (layout == CblasRowMajor) rowMajorGemvTrans(trans) else transChar(trans);
    var aa = asBlasInt(lda);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.dgemv_(&tt, &mm, &nn, &alpha, a, &aa, x, &ix, &beta, y, &iy);
}

pub export fn cblas_cgemv(layout: c_int, trans: c_int, m: c_int, n: c_int, alpha: *const ComplexF32, a: [*]const ComplexF32, lda: c_int, x: [*]const ComplexF32, incx: c_int, beta: *const ComplexF32, y: [*]ComplexF32, incy: c_int) callconv(.c) void {
    if (!validLayout(layout)) return;
    if (layout == CblasRowMajor and trans == CblasConjTrans) {
        rowMajorConjGemv(ComplexF32, asBlasInt(m), asBlasInt(n), alpha.*, a, asBlasInt(lda), x, asBlasInt(incx), beta.*, y, asBlasInt(incy));
        return;
    }
    var mm = asBlasInt(if (layout == CblasRowMajor) n else m);
    var nn = asBlasInt(if (layout == CblasRowMajor) m else n);
    var tt = if (layout == CblasRowMajor) rowMajorGemvTrans(trans) else transChar(trans);
    var aa = asBlasInt(lda);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.cgemv_(&tt, &mm, &nn, alpha, a, &aa, x, &ix, beta, y, &iy);
}

pub export fn cblas_zgemv(layout: c_int, trans: c_int, m: c_int, n: c_int, alpha: *const ComplexF64, a: [*]const ComplexF64, lda: c_int, x: [*]const ComplexF64, incx: c_int, beta: *const ComplexF64, y: [*]ComplexF64, incy: c_int) callconv(.c) void {
    if (!validLayout(layout)) return;
    if (layout == CblasRowMajor and trans == CblasConjTrans) {
        rowMajorConjGemv(ComplexF64, asBlasInt(m), asBlasInt(n), alpha.*, a, asBlasInt(lda), x, asBlasInt(incx), beta.*, y, asBlasInt(incy));
        return;
    }
    var mm = asBlasInt(if (layout == CblasRowMajor) n else m);
    var nn = asBlasInt(if (layout == CblasRowMajor) m else n);
    var tt = if (layout == CblasRowMajor) rowMajorGemvTrans(trans) else transChar(trans);
    var aa = asBlasInt(lda);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.zgemv_(&tt, &mm, &nn, alpha, a, &aa, x, &ix, beta, y, &iy);
}

pub export fn cblas_sgbmv(layout: c_int, trans: c_int, m: c_int, n: c_int, kl: c_int, ku: c_int, alpha: f32, a: [*]const f32, lda: c_int, x: [*]const f32, incx: c_int, beta: f32, y: [*]f32, incy: c_int) callconv(.c) void {
    if (!validLayout(layout)) return;
    var mm = asBlasInt(if (layout == CblasRowMajor) n else m);
    var nn = asBlasInt(if (layout == CblasRowMajor) m else n);
    var kll = asBlasInt(if (layout == CblasRowMajor) ku else kl);
    var kuu = asBlasInt(if (layout == CblasRowMajor) kl else ku);
    var tt = if (layout == CblasRowMajor) rowMajorGemvTrans(trans) else transChar(trans);
    var aa = asBlasInt(lda);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.sgbmv_(&tt, &mm, &nn, &kll, &kuu, &alpha, a, &aa, x, &ix, &beta, y, &iy);
}

pub export fn cblas_dgbmv(layout: c_int, trans: c_int, m: c_int, n: c_int, kl: c_int, ku: c_int, alpha: f64, a: [*]const f64, lda: c_int, x: [*]const f64, incx: c_int, beta: f64, y: [*]f64, incy: c_int) callconv(.c) void {
    if (!validLayout(layout)) return;
    var mm = asBlasInt(if (layout == CblasRowMajor) n else m);
    var nn = asBlasInt(if (layout == CblasRowMajor) m else n);
    var kll = asBlasInt(if (layout == CblasRowMajor) ku else kl);
    var kuu = asBlasInt(if (layout == CblasRowMajor) kl else ku);
    var tt = if (layout == CblasRowMajor) rowMajorGemvTrans(trans) else transChar(trans);
    var aa = asBlasInt(lda);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.dgbmv_(&tt, &mm, &nn, &kll, &kuu, &alpha, a, &aa, x, &ix, &beta, y, &iy);
}

pub export fn cblas_cgbmv(layout: c_int, trans: c_int, m: c_int, n: c_int, kl: c_int, ku: c_int, alpha: *const ComplexF32, a: [*]const ComplexF32, lda: c_int, x: [*]const ComplexF32, incx: c_int, beta: *const ComplexF32, y: [*]ComplexF32, incy: c_int) callconv(.c) void {
    if (!validLayout(layout)) return;
    var aa = asBlasInt(lda);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    if (layout == CblasRowMajor and trans == CblasConjTrans) {
        rowMajorConjGbmv(ComplexF32, asBlasInt(m), asBlasInt(n), asBlasInt(kl), asBlasInt(ku), alpha.*, a, aa, x, ix, beta.*, y, iy);
        return;
    }
    var mm = asBlasInt(if (layout == CblasRowMajor) n else m);
    var nn = asBlasInt(if (layout == CblasRowMajor) m else n);
    var kll = asBlasInt(if (layout == CblasRowMajor) ku else kl);
    var kuu = asBlasInt(if (layout == CblasRowMajor) kl else ku);
    var tt = if (layout == CblasRowMajor) rowMajorGemvTrans(trans) else transChar(trans);
    f.cgbmv_(&tt, &mm, &nn, &kll, &kuu, alpha, a, &aa, x, &ix, beta, y, &iy);
}

pub export fn cblas_zgbmv(layout: c_int, trans: c_int, m: c_int, n: c_int, kl: c_int, ku: c_int, alpha: *const ComplexF64, a: [*]const ComplexF64, lda: c_int, x: [*]const ComplexF64, incx: c_int, beta: *const ComplexF64, y: [*]ComplexF64, incy: c_int) callconv(.c) void {
    if (!validLayout(layout)) return;
    var aa = asBlasInt(lda);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    if (layout == CblasRowMajor and trans == CblasConjTrans) {
        rowMajorConjGbmv(ComplexF64, asBlasInt(m), asBlasInt(n), asBlasInt(kl), asBlasInt(ku), alpha.*, a, aa, x, ix, beta.*, y, iy);
        return;
    }
    var mm = asBlasInt(if (layout == CblasRowMajor) n else m);
    var nn = asBlasInt(if (layout == CblasRowMajor) m else n);
    var kll = asBlasInt(if (layout == CblasRowMajor) ku else kl);
    var kuu = asBlasInt(if (layout == CblasRowMajor) kl else ku);
    var tt = if (layout == CblasRowMajor) rowMajorGemvTrans(trans) else transChar(trans);
    f.zgbmv_(&tt, &mm, &nn, &kll, &kuu, alpha, a, &aa, x, &ix, beta, y, &iy);
}

pub export fn cblas_ssymv(layout: c_int, uplo: c_int, n: c_int, alpha: f32, a: [*]const f32, lda: c_int, x: [*]const f32, incx: c_int, beta: f32, y: [*]f32, incy: c_int) callconv(.c) void {
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var nn = asBlasInt(n);
    var aa = asBlasInt(lda);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.ssymv_(&uu, &nn, &alpha, a, &aa, x, &ix, &beta, y, &iy);
}
pub export fn cblas_dsymv(layout: c_int, uplo: c_int, n: c_int, alpha: f64, a: [*]const f64, lda: c_int, x: [*]const f64, incx: c_int, beta: f64, y: [*]f64, incy: c_int) callconv(.c) void {
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var nn = asBlasInt(n);
    var aa = asBlasInt(lda);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.dsymv_(&uu, &nn, &alpha, a, &aa, x, &ix, &beta, y, &iy);
}
pub export fn cblas_chemv(layout: c_int, uplo: c_int, n: c_int, alpha: *const ComplexF32, a: [*]const ComplexF32, lda: c_int, x: [*]const ComplexF32, incx: c_int, beta: *const ComplexF32, y: [*]ComplexF32, incy: c_int) callconv(.c) void {
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var nn = asBlasInt(n);
    var aa = asBlasInt(lda);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.chemv_(&uu, &nn, alpha, a, &aa, x, &ix, beta, y, &iy);
}
pub export fn cblas_zhemv(layout: c_int, uplo: c_int, n: c_int, alpha: *const ComplexF64, a: [*]const ComplexF64, lda: c_int, x: [*]const ComplexF64, incx: c_int, beta: *const ComplexF64, y: [*]ComplexF64, incy: c_int) callconv(.c) void {
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var nn = asBlasInt(n);
    var aa = asBlasInt(lda);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.zhemv_(&uu, &nn, alpha, a, &aa, x, &ix, beta, y, &iy);
}

pub export fn cblas_ssbmv(layout: c_int, uplo: c_int, n: c_int, k: c_int, alpha: f32, a: [*]const f32, lda: c_int, x: [*]const f32, incx: c_int, beta: f32, y: [*]f32, incy: c_int) callconv(.c) void {
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var nn = asBlasInt(n);
    var kk = asBlasInt(k);
    var aa = asBlasInt(lda);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.ssbmv_(&uu, &nn, &kk, &alpha, a, &aa, x, &ix, &beta, y, &iy);
}
pub export fn cblas_dsbmv(layout: c_int, uplo: c_int, n: c_int, k: c_int, alpha: f64, a: [*]const f64, lda: c_int, x: [*]const f64, incx: c_int, beta: f64, y: [*]f64, incy: c_int) callconv(.c) void {
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var nn = asBlasInt(n);
    var kk = asBlasInt(k);
    var aa = asBlasInt(lda);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.dsbmv_(&uu, &nn, &kk, &alpha, a, &aa, x, &ix, &beta, y, &iy);
}
pub export fn cblas_chbmv(layout: c_int, uplo: c_int, n: c_int, k: c_int, alpha: *const ComplexF32, a: [*]const ComplexF32, lda: c_int, x: [*]const ComplexF32, incx: c_int, beta: *const ComplexF32, y: [*]ComplexF32, incy: c_int) callconv(.c) void {
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var nn = asBlasInt(n);
    var kk = asBlasInt(k);
    var aa = asBlasInt(lda);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.chbmv_(&uu, &nn, &kk, alpha, a, &aa, x, &ix, beta, y, &iy);
}
pub export fn cblas_zhbmv(layout: c_int, uplo: c_int, n: c_int, k: c_int, alpha: *const ComplexF64, a: [*]const ComplexF64, lda: c_int, x: [*]const ComplexF64, incx: c_int, beta: *const ComplexF64, y: [*]ComplexF64, incy: c_int) callconv(.c) void {
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var nn = asBlasInt(n);
    var kk = asBlasInt(k);
    var aa = asBlasInt(lda);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.zhbmv_(&uu, &nn, &kk, alpha, a, &aa, x, &ix, beta, y, &iy);
}

pub export fn cblas_sspmv(layout: c_int, uplo: c_int, n: c_int, alpha: f32, ap: [*]const f32, x: [*]const f32, incx: c_int, beta: f32, y: [*]f32, incy: c_int) callconv(.c) void {
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.sspmv_(&uu, &nn, &alpha, ap, x, &ix, &beta, y, &iy);
}
pub export fn cblas_dspmv(layout: c_int, uplo: c_int, n: c_int, alpha: f64, ap: [*]const f64, x: [*]const f64, incx: c_int, beta: f64, y: [*]f64, incy: c_int) callconv(.c) void {
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.dspmv_(&uu, &nn, &alpha, ap, x, &ix, &beta, y, &iy);
}
pub export fn cblas_chpmv(layout: c_int, uplo: c_int, n: c_int, alpha: *const ComplexF32, ap: [*]const ComplexF32, x: [*]const ComplexF32, incx: c_int, beta: *const ComplexF32, y: [*]ComplexF32, incy: c_int) callconv(.c) void {
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.chpmv_(&uu, &nn, alpha, ap, x, &ix, beta, y, &iy);
}
pub export fn cblas_zhpmv(layout: c_int, uplo: c_int, n: c_int, alpha: *const ComplexF64, ap: [*]const ComplexF64, x: [*]const ComplexF64, incx: c_int, beta: *const ComplexF64, y: [*]ComplexF64, incy: c_int) callconv(.c) void {
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.zhpmv_(&uu, &nn, alpha, ap, x, &ix, beta, y, &iy);
}

pub export fn cblas_strmv(layout: c_int, uplo: c_int, trans: c_int, diag: c_int, n: c_int, a: [*]const f32, lda: c_int, x: [*]f32, incx: c_int) callconv(.c) void {
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var tt = if (layout == CblasRowMajor) rowMajorTransChar(trans) else transChar(trans);
    var dd = diagChar(diag);
    var nn = asBlasInt(n);
    var aa = asBlasInt(lda);
    var ix = asBlasInt(incx);
    f.strmv_(&uu, &tt, &dd, &nn, a, &aa, x, &ix);
}
pub export fn cblas_dtrmv(layout: c_int, uplo: c_int, trans: c_int, diag: c_int, n: c_int, a: [*]const f64, lda: c_int, x: [*]f64, incx: c_int) callconv(.c) void {
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var tt = if (layout == CblasRowMajor) rowMajorTransChar(trans) else transChar(trans);
    var dd = diagChar(diag);
    var nn = asBlasInt(n);
    var aa = asBlasInt(lda);
    var ix = asBlasInt(incx);
    f.dtrmv_(&uu, &tt, &dd, &nn, a, &aa, x, &ix);
}
pub export fn cblas_ctrmv(layout: c_int, uplo: c_int, trans: c_int, diag: c_int, n: c_int, a: [*]const ComplexF32, lda: c_int, x: [*]ComplexF32, incx: c_int) callconv(.c) void {
    if (!validLayout(layout)) return;
    var nn = asBlasInt(n);
    var aa = asBlasInt(lda);
    var ix = asBlasInt(incx);
    if (layout == CblasRowMajor and trans == CblasConjTrans) {
        rowMajorConjTrmv(ComplexF32, uplo, diag, nn, a, aa, x, ix);
        return;
    }
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var tt = if (layout == CblasRowMajor) rowMajorTransChar(trans) else transChar(trans);
    var dd = diagChar(diag);
    f.ctrmv_(&uu, &tt, &dd, &nn, a, &aa, x, &ix);
}
pub export fn cblas_ztrmv(layout: c_int, uplo: c_int, trans: c_int, diag: c_int, n: c_int, a: [*]const ComplexF64, lda: c_int, x: [*]ComplexF64, incx: c_int) callconv(.c) void {
    if (!validLayout(layout)) return;
    var nn = asBlasInt(n);
    var aa = asBlasInt(lda);
    var ix = asBlasInt(incx);
    if (layout == CblasRowMajor and trans == CblasConjTrans) {
        rowMajorConjTrmv(ComplexF64, uplo, diag, nn, a, aa, x, ix);
        return;
    }
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var tt = if (layout == CblasRowMajor) rowMajorTransChar(trans) else transChar(trans);
    var dd = diagChar(diag);
    f.ztrmv_(&uu, &tt, &dd, &nn, a, &aa, x, &ix);
}

pub export fn cblas_stbmv(layout: c_int, uplo: c_int, trans: c_int, diag: c_int, n: c_int, k: c_int, a: [*]const f32, lda: c_int, x: [*]f32, incx: c_int) callconv(.c) void {
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var tt = if (layout == CblasRowMajor) rowMajorTransChar(trans) else transChar(trans);
    var dd = diagChar(diag);
    var nn = asBlasInt(n);
    var kk = asBlasInt(k);
    var aa = asBlasInt(lda);
    var ix = asBlasInt(incx);
    f.stbmv_(&uu, &tt, &dd, &nn, &kk, a, &aa, x, &ix);
}
pub export fn cblas_dtbmv(layout: c_int, uplo: c_int, trans: c_int, diag: c_int, n: c_int, k: c_int, a: [*]const f64, lda: c_int, x: [*]f64, incx: c_int) callconv(.c) void {
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var tt = if (layout == CblasRowMajor) rowMajorTransChar(trans) else transChar(trans);
    var dd = diagChar(diag);
    var nn = asBlasInt(n);
    var kk = asBlasInt(k);
    var aa = asBlasInt(lda);
    var ix = asBlasInt(incx);
    f.dtbmv_(&uu, &tt, &dd, &nn, &kk, a, &aa, x, &ix);
}
pub export fn cblas_ctbmv(layout: c_int, uplo: c_int, trans: c_int, diag: c_int, n: c_int, k: c_int, a: [*]const ComplexF32, lda: c_int, x: [*]ComplexF32, incx: c_int) callconv(.c) void {
    if (!validLayout(layout)) return;
    var nn = asBlasInt(n);
    var kk = asBlasInt(k);
    var aa = asBlasInt(lda);
    var ix = asBlasInt(incx);
    if (layout == CblasRowMajor and trans == CblasConjTrans) {
        rowMajorConjTbmv(ComplexF32, uplo, diag, nn, kk, a, aa, x, ix);
        return;
    }
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var tt = if (layout == CblasRowMajor) rowMajorTransChar(trans) else transChar(trans);
    var dd = diagChar(diag);
    f.ctbmv_(&uu, &tt, &dd, &nn, &kk, a, &aa, x, &ix);
}
pub export fn cblas_ztbmv(layout: c_int, uplo: c_int, trans: c_int, diag: c_int, n: c_int, k: c_int, a: [*]const ComplexF64, lda: c_int, x: [*]ComplexF64, incx: c_int) callconv(.c) void {
    if (!validLayout(layout)) return;
    var nn = asBlasInt(n);
    var kk = asBlasInt(k);
    var aa = asBlasInt(lda);
    var ix = asBlasInt(incx);
    if (layout == CblasRowMajor and trans == CblasConjTrans) {
        rowMajorConjTbmv(ComplexF64, uplo, diag, nn, kk, a, aa, x, ix);
        return;
    }
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var tt = if (layout == CblasRowMajor) rowMajorTransChar(trans) else transChar(trans);
    var dd = diagChar(diag);
    f.ztbmv_(&uu, &tt, &dd, &nn, &kk, a, &aa, x, &ix);
}

pub export fn cblas_stpmv(layout: c_int, uplo: c_int, trans: c_int, diag: c_int, n: c_int, ap: [*]const f32, x: [*]f32, incx: c_int) callconv(.c) void {
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var tt = if (layout == CblasRowMajor) rowMajorTransChar(trans) else transChar(trans);
    var dd = diagChar(diag);
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    f.stpmv_(&uu, &tt, &dd, &nn, ap, x, &ix);
}
pub export fn cblas_dtpmv(layout: c_int, uplo: c_int, trans: c_int, diag: c_int, n: c_int, ap: [*]const f64, x: [*]f64, incx: c_int) callconv(.c) void {
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var tt = if (layout == CblasRowMajor) rowMajorTransChar(trans) else transChar(trans);
    var dd = diagChar(diag);
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    f.dtpmv_(&uu, &tt, &dd, &nn, ap, x, &ix);
}
pub export fn cblas_ctpmv(layout: c_int, uplo: c_int, trans: c_int, diag: c_int, n: c_int, ap: [*]const ComplexF32, x: [*]ComplexF32, incx: c_int) callconv(.c) void {
    if (!validLayout(layout)) return;
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    if (layout == CblasRowMajor and trans == CblasConjTrans) {
        rowMajorConjTpmv(ComplexF32, uplo, diag, nn, ap, x, ix);
        return;
    }
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var tt = if (layout == CblasRowMajor) rowMajorTransChar(trans) else transChar(trans);
    var dd = diagChar(diag);
    f.ctpmv_(&uu, &tt, &dd, &nn, ap, x, &ix);
}
pub export fn cblas_ztpmv(layout: c_int, uplo: c_int, trans: c_int, diag: c_int, n: c_int, ap: [*]const ComplexF64, x: [*]ComplexF64, incx: c_int) callconv(.c) void {
    if (!validLayout(layout)) return;
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    if (layout == CblasRowMajor and trans == CblasConjTrans) {
        rowMajorConjTpmv(ComplexF64, uplo, diag, nn, ap, x, ix);
        return;
    }
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var tt = if (layout == CblasRowMajor) rowMajorTransChar(trans) else transChar(trans);
    var dd = diagChar(diag);
    f.ztpmv_(&uu, &tt, &dd, &nn, ap, x, &ix);
}

pub export fn cblas_strsv(layout: c_int, uplo: c_int, trans: c_int, diag: c_int, n: c_int, a: [*]const f32, lda: c_int, x: [*]f32, incx: c_int) callconv(.c) void {
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var tt = if (layout == CblasRowMajor) rowMajorTransChar(trans) else transChar(trans);
    var dd = diagChar(diag);
    var nn = asBlasInt(n);
    var aa = asBlasInt(lda);
    var ix = asBlasInt(incx);
    f.strsv_(&uu, &tt, &dd, &nn, a, &aa, x, &ix);
}
pub export fn cblas_dtrsv(layout: c_int, uplo: c_int, trans: c_int, diag: c_int, n: c_int, a: [*]const f64, lda: c_int, x: [*]f64, incx: c_int) callconv(.c) void {
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var tt = if (layout == CblasRowMajor) rowMajorTransChar(trans) else transChar(trans);
    var dd = diagChar(diag);
    var nn = asBlasInt(n);
    var aa = asBlasInt(lda);
    var ix = asBlasInt(incx);
    f.dtrsv_(&uu, &tt, &dd, &nn, a, &aa, x, &ix);
}
pub export fn cblas_ctrsv(layout: c_int, uplo: c_int, trans: c_int, diag: c_int, n: c_int, a: [*]const ComplexF32, lda: c_int, x: [*]ComplexF32, incx: c_int) callconv(.c) void {
    if (!validLayout(layout)) return;
    var nn = asBlasInt(n);
    var aa = asBlasInt(lda);
    var ix = asBlasInt(incx);
    if (layout == CblasRowMajor and trans == CblasConjTrans) {
        rowMajorConjTrsv(ComplexF32, uplo, diag, nn, a, aa, x, ix);
        return;
    }
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var tt = if (layout == CblasRowMajor) rowMajorTransChar(trans) else transChar(trans);
    var dd = diagChar(diag);
    f.ctrsv_(&uu, &tt, &dd, &nn, a, &aa, x, &ix);
}
pub export fn cblas_ztrsv(layout: c_int, uplo: c_int, trans: c_int, diag: c_int, n: c_int, a: [*]const ComplexF64, lda: c_int, x: [*]ComplexF64, incx: c_int) callconv(.c) void {
    if (!validLayout(layout)) return;
    var nn = asBlasInt(n);
    var aa = asBlasInt(lda);
    var ix = asBlasInt(incx);
    if (layout == CblasRowMajor and trans == CblasConjTrans) {
        rowMajorConjTrsv(ComplexF64, uplo, diag, nn, a, aa, x, ix);
        return;
    }
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var tt = if (layout == CblasRowMajor) rowMajorTransChar(trans) else transChar(trans);
    var dd = diagChar(diag);
    f.ztrsv_(&uu, &tt, &dd, &nn, a, &aa, x, &ix);
}

pub export fn cblas_stbsv(layout: c_int, uplo: c_int, trans: c_int, diag: c_int, n: c_int, k: c_int, a: [*]const f32, lda: c_int, x: [*]f32, incx: c_int) callconv(.c) void {
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var tt = if (layout == CblasRowMajor) rowMajorTransChar(trans) else transChar(trans);
    var dd = diagChar(diag);
    var nn = asBlasInt(n);
    var kk = asBlasInt(k);
    var aa = asBlasInt(lda);
    var ix = asBlasInt(incx);
    f.stbsv_(&uu, &tt, &dd, &nn, &kk, a, &aa, x, &ix);
}
pub export fn cblas_dtbsv(layout: c_int, uplo: c_int, trans: c_int, diag: c_int, n: c_int, k: c_int, a: [*]const f64, lda: c_int, x: [*]f64, incx: c_int) callconv(.c) void {
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var tt = if (layout == CblasRowMajor) rowMajorTransChar(trans) else transChar(trans);
    var dd = diagChar(diag);
    var nn = asBlasInt(n);
    var kk = asBlasInt(k);
    var aa = asBlasInt(lda);
    var ix = asBlasInt(incx);
    f.dtbsv_(&uu, &tt, &dd, &nn, &kk, a, &aa, x, &ix);
}
pub export fn cblas_ctbsv(layout: c_int, uplo: c_int, trans: c_int, diag: c_int, n: c_int, k: c_int, a: [*]const ComplexF32, lda: c_int, x: [*]ComplexF32, incx: c_int) callconv(.c) void {
    if (!validLayout(layout)) return;
    var nn = asBlasInt(n);
    var kk = asBlasInt(k);
    var aa = asBlasInt(lda);
    var ix = asBlasInt(incx);
    if (layout == CblasRowMajor and trans == CblasConjTrans) {
        rowMajorConjTbsv(ComplexF32, uplo, diag, nn, kk, a, aa, x, ix);
        return;
    }
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var tt = if (layout == CblasRowMajor) rowMajorTransChar(trans) else transChar(trans);
    var dd = diagChar(diag);
    f.ctbsv_(&uu, &tt, &dd, &nn, &kk, a, &aa, x, &ix);
}
pub export fn cblas_ztbsv(layout: c_int, uplo: c_int, trans: c_int, diag: c_int, n: c_int, k: c_int, a: [*]const ComplexF64, lda: c_int, x: [*]ComplexF64, incx: c_int) callconv(.c) void {
    if (!validLayout(layout)) return;
    var nn = asBlasInt(n);
    var kk = asBlasInt(k);
    var aa = asBlasInt(lda);
    var ix = asBlasInt(incx);
    if (layout == CblasRowMajor and trans == CblasConjTrans) {
        rowMajorConjTbsv(ComplexF64, uplo, diag, nn, kk, a, aa, x, ix);
        return;
    }
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var tt = if (layout == CblasRowMajor) rowMajorTransChar(trans) else transChar(trans);
    var dd = diagChar(diag);
    f.ztbsv_(&uu, &tt, &dd, &nn, &kk, a, &aa, x, &ix);
}

pub export fn cblas_stpsv(layout: c_int, uplo: c_int, trans: c_int, diag: c_int, n: c_int, ap: [*]const f32, x: [*]f32, incx: c_int) callconv(.c) void {
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var tt = if (layout == CblasRowMajor) rowMajorTransChar(trans) else transChar(trans);
    var dd = diagChar(diag);
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    f.stpsv_(&uu, &tt, &dd, &nn, ap, x, &ix);
}
pub export fn cblas_dtpsv(layout: c_int, uplo: c_int, trans: c_int, diag: c_int, n: c_int, ap: [*]const f64, x: [*]f64, incx: c_int) callconv(.c) void {
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var tt = if (layout == CblasRowMajor) rowMajorTransChar(trans) else transChar(trans);
    var dd = diagChar(diag);
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    f.dtpsv_(&uu, &tt, &dd, &nn, ap, x, &ix);
}
pub export fn cblas_ctpsv(layout: c_int, uplo: c_int, trans: c_int, diag: c_int, n: c_int, ap: [*]const ComplexF32, x: [*]ComplexF32, incx: c_int) callconv(.c) void {
    if (!validLayout(layout)) return;
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    if (layout == CblasRowMajor and trans == CblasConjTrans) {
        rowMajorConjTpsv(ComplexF32, uplo, diag, nn, ap, x, ix);
        return;
    }
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var tt = if (layout == CblasRowMajor) rowMajorTransChar(trans) else transChar(trans);
    var dd = diagChar(diag);
    f.ctpsv_(&uu, &tt, &dd, &nn, ap, x, &ix);
}
pub export fn cblas_ztpsv(layout: c_int, uplo: c_int, trans: c_int, diag: c_int, n: c_int, ap: [*]const ComplexF64, x: [*]ComplexF64, incx: c_int) callconv(.c) void {
    if (!validLayout(layout)) return;
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    if (layout == CblasRowMajor and trans == CblasConjTrans) {
        rowMajorConjTpsv(ComplexF64, uplo, diag, nn, ap, x, ix);
        return;
    }
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var tt = if (layout == CblasRowMajor) rowMajorTransChar(trans) else transChar(trans);
    var dd = diagChar(diag);
    f.ztpsv_(&uu, &tt, &dd, &nn, ap, x, &ix);
}

pub export fn cblas_sger(layout: c_int, m: c_int, n: c_int, alpha: f32, x: [*]const f32, incx: c_int, y: [*]const f32, incy: c_int, a: [*]f32, lda: c_int) callconv(.c) void {
    if (!validLayout(layout)) return;
    var mm = asBlasInt(m);
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    var aa = asBlasInt(lda);
    if (layout == CblasRowMajor) rowMajorGer(f32, mm, nn, alpha, x, ix, y, iy, a, aa, false) else f.sger_(&mm, &nn, &alpha, x, &ix, y, &iy, a, &aa);
}

pub export fn cblas_dger(layout: c_int, m: c_int, n: c_int, alpha: f64, x: [*]const f64, incx: c_int, y: [*]const f64, incy: c_int, a: [*]f64, lda: c_int) callconv(.c) void {
    if (!validLayout(layout)) return;
    var mm = asBlasInt(m);
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    var aa = asBlasInt(lda);
    if (layout == CblasRowMajor) rowMajorGer(f64, mm, nn, alpha, x, ix, y, iy, a, aa, false) else f.dger_(&mm, &nn, &alpha, x, &ix, y, &iy, a, &aa);
}

pub export fn cblas_cgeru(layout: c_int, m: c_int, n: c_int, alpha: *const ComplexF32, x: [*]const ComplexF32, incx: c_int, y: [*]const ComplexF32, incy: c_int, a: [*]ComplexF32, lda: c_int) callconv(.c) void {
    if (!validLayout(layout)) return;
    var mm = asBlasInt(m);
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    var aa = asBlasInt(lda);
    if (layout == CblasRowMajor) rowMajorGer(ComplexF32, mm, nn, alpha.*, x, ix, y, iy, a, aa, false) else f.cgeru_(&mm, &nn, alpha, x, &ix, y, &iy, a, &aa);
}

pub export fn cblas_zgeru(layout: c_int, m: c_int, n: c_int, alpha: *const ComplexF64, x: [*]const ComplexF64, incx: c_int, y: [*]const ComplexF64, incy: c_int, a: [*]ComplexF64, lda: c_int) callconv(.c) void {
    if (!validLayout(layout)) return;
    var mm = asBlasInt(m);
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    var aa = asBlasInt(lda);
    if (layout == CblasRowMajor) rowMajorGer(ComplexF64, mm, nn, alpha.*, x, ix, y, iy, a, aa, false) else f.zgeru_(&mm, &nn, alpha, x, &ix, y, &iy, a, &aa);
}

pub export fn cblas_cgerc(layout: c_int, m: c_int, n: c_int, alpha: *const ComplexF32, x: [*]const ComplexF32, incx: c_int, y: [*]const ComplexF32, incy: c_int, a: [*]ComplexF32, lda: c_int) callconv(.c) void {
    if (!validLayout(layout)) return;
    var mm = asBlasInt(m);
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    var aa = asBlasInt(lda);
    if (layout == CblasRowMajor) rowMajorGer(ComplexF32, mm, nn, alpha.*, x, ix, y, iy, a, aa, true) else f.cgerc_(&mm, &nn, alpha, x, &ix, y, &iy, a, &aa);
}

pub export fn cblas_zgerc(layout: c_int, m: c_int, n: c_int, alpha: *const ComplexF64, x: [*]const ComplexF64, incx: c_int, y: [*]const ComplexF64, incy: c_int, a: [*]ComplexF64, lda: c_int) callconv(.c) void {
    if (!validLayout(layout)) return;
    var mm = asBlasInt(m);
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    var aa = asBlasInt(lda);
    if (layout == CblasRowMajor) rowMajorGer(ComplexF64, mm, nn, alpha.*, x, ix, y, iy, a, aa, true) else f.zgerc_(&mm, &nn, alpha, x, &ix, y, &iy, a, &aa);
}

pub export fn cblas_ssyr(layout: c_int, uplo: c_int, n: c_int, alpha: f32, x: [*]const f32, incx: c_int, a: [*]f32, lda: c_int) callconv(.c) void {
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var aa = asBlasInt(lda);
    f.ssyr_(&uu, &nn, &alpha, x, &ix, a, &aa);
}
pub export fn cblas_dsyr(layout: c_int, uplo: c_int, n: c_int, alpha: f64, x: [*]const f64, incx: c_int, a: [*]f64, lda: c_int) callconv(.c) void {
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var aa = asBlasInt(lda);
    f.dsyr_(&uu, &nn, &alpha, x, &ix, a, &aa);
}
pub export fn cblas_cher(layout: c_int, uplo: c_int, n: c_int, alpha: f32, x: [*]const ComplexF32, incx: c_int, a: [*]ComplexF32, lda: c_int) callconv(.c) void {
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var aa = asBlasInt(lda);
    f.cher_(&uu, &nn, &alpha, x, &ix, a, &aa);
}
pub export fn cblas_zher(layout: c_int, uplo: c_int, n: c_int, alpha: f64, x: [*]const ComplexF64, incx: c_int, a: [*]ComplexF64, lda: c_int) callconv(.c) void {
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var aa = asBlasInt(lda);
    f.zher_(&uu, &nn, &alpha, x, &ix, a, &aa);
}

pub export fn cblas_sspr(layout: c_int, uplo: c_int, n: c_int, alpha: f32, x: [*]const f32, incx: c_int, ap: [*]f32) callconv(.c) void {
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    f.sspr_(&uu, &nn, &alpha, x, &ix, ap);
}
pub export fn cblas_dspr(layout: c_int, uplo: c_int, n: c_int, alpha: f64, x: [*]const f64, incx: c_int, ap: [*]f64) callconv(.c) void {
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    f.dspr_(&uu, &nn, &alpha, x, &ix, ap);
}
pub export fn cblas_chpr(layout: c_int, uplo: c_int, n: c_int, alpha: f32, x: [*]const ComplexF32, incx: c_int, ap: [*]ComplexF32) callconv(.c) void {
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    f.chpr_(&uu, &nn, &alpha, x, &ix, ap);
}
pub export fn cblas_zhpr(layout: c_int, uplo: c_int, n: c_int, alpha: f64, x: [*]const ComplexF64, incx: c_int, ap: [*]ComplexF64) callconv(.c) void {
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    f.zhpr_(&uu, &nn, &alpha, x, &ix, ap);
}

pub export fn cblas_ssyr2(layout: c_int, uplo: c_int, n: c_int, alpha: f32, x: [*]const f32, incx: c_int, y: [*]const f32, incy: c_int, a: [*]f32, lda: c_int) callconv(.c) void {
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    var aa = asBlasInt(lda);
    f.ssyr2_(&uu, &nn, &alpha, x, &ix, y, &iy, a, &aa);
}
pub export fn cblas_dsyr2(layout: c_int, uplo: c_int, n: c_int, alpha: f64, x: [*]const f64, incx: c_int, y: [*]const f64, incy: c_int, a: [*]f64, lda: c_int) callconv(.c) void {
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    var aa = asBlasInt(lda);
    f.dsyr2_(&uu, &nn, &alpha, x, &ix, y, &iy, a, &aa);
}
pub export fn cblas_cher2(layout: c_int, uplo: c_int, n: c_int, alpha: *const ComplexF32, x: [*]const ComplexF32, incx: c_int, y: [*]const ComplexF32, incy: c_int, a: [*]ComplexF32, lda: c_int) callconv(.c) void {
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    var aa = asBlasInt(lda);
    f.cher2_(&uu, &nn, alpha, x, &ix, y, &iy, a, &aa);
}
pub export fn cblas_zher2(layout: c_int, uplo: c_int, n: c_int, alpha: *const ComplexF64, x: [*]const ComplexF64, incx: c_int, y: [*]const ComplexF64, incy: c_int, a: [*]ComplexF64, lda: c_int) callconv(.c) void {
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    var aa = asBlasInt(lda);
    f.zher2_(&uu, &nn, alpha, x, &ix, y, &iy, a, &aa);
}

pub export fn cblas_sspr2(layout: c_int, uplo: c_int, n: c_int, alpha: f32, x: [*]const f32, incx: c_int, y: [*]const f32, incy: c_int, ap: [*]f32) callconv(.c) void {
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.sspr2_(&uu, &nn, &alpha, x, &ix, y, &iy, ap);
}
pub export fn cblas_dspr2(layout: c_int, uplo: c_int, n: c_int, alpha: f64, x: [*]const f64, incx: c_int, y: [*]const f64, incy: c_int, ap: [*]f64) callconv(.c) void {
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.dspr2_(&uu, &nn, &alpha, x, &ix, y, &iy, ap);
}
pub export fn cblas_chpr2(layout: c_int, uplo: c_int, n: c_int, alpha: *const ComplexF32, x: [*]const ComplexF32, incx: c_int, y: [*]const ComplexF32, incy: c_int, ap: [*]ComplexF32) callconv(.c) void {
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.chpr2_(&uu, &nn, alpha, x, &ix, y, &iy, ap);
}
pub export fn cblas_zhpr2(layout: c_int, uplo: c_int, n: c_int, alpha: *const ComplexF64, x: [*]const ComplexF64, incx: c_int, y: [*]const ComplexF64, incy: c_int, ap: [*]ComplexF64) callconv(.c) void {
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.zhpr2_(&uu, &nn, alpha, x, &ix, y, &iy, ap);
}

// Level 3 CBLAS wrappers.
pub export fn cblas_sgemm(layout: c_int, transa: c_int, transb: c_int, m: c_int, n: c_int, k: c_int, alpha: f32, a: [*]const f32, lda: c_int, b: [*]const f32, ldb: c_int, beta: f32, c: [*]f32, ldc: c_int) callconv(.c) void {
    if (!validLayout(layout) or !validTrans(transa) or !validTrans(transb)) return;
    var ta = if (layout == CblasRowMajor) transChar(transb) else transChar(transa);
    var tb = if (layout == CblasRowMajor) transChar(transa) else transChar(transb);
    var mm = asBlasInt(if (layout == CblasRowMajor) n else m);
    var nn = asBlasInt(if (layout == CblasRowMajor) m else n);
    var kk = asBlasInt(k);
    var la = asBlasInt(if (layout == CblasRowMajor) ldb else lda);
    var lb = asBlasInt(if (layout == CblasRowMajor) lda else ldb);
    var lc = asBlasInt(ldc);
    if (layout == CblasRowMajor) f.sgemm_(&ta, &tb, &mm, &nn, &kk, &alpha, b, &la, a, &lb, &beta, c, &lc) else f.sgemm_(&ta, &tb, &mm, &nn, &kk, &alpha, a, &la, b, &lb, &beta, c, &lc);
}

pub export fn cblas_dgemm(layout: c_int, transa: c_int, transb: c_int, m: c_int, n: c_int, k: c_int, alpha: f64, a: [*]const f64, lda: c_int, b: [*]const f64, ldb: c_int, beta: f64, c: [*]f64, ldc: c_int) callconv(.c) void {
    if (!validLayout(layout) or !validTrans(transa) or !validTrans(transb)) return;
    var ta = if (layout == CblasRowMajor) transChar(transb) else transChar(transa);
    var tb = if (layout == CblasRowMajor) transChar(transa) else transChar(transb);
    var mm = asBlasInt(if (layout == CblasRowMajor) n else m);
    var nn = asBlasInt(if (layout == CblasRowMajor) m else n);
    var kk = asBlasInt(k);
    var la = asBlasInt(if (layout == CblasRowMajor) ldb else lda);
    var lb = asBlasInt(if (layout == CblasRowMajor) lda else ldb);
    var lc = asBlasInt(ldc);
    if (layout == CblasRowMajor) f.dgemm_(&ta, &tb, &mm, &nn, &kk, &alpha, b, &la, a, &lb, &beta, c, &lc) else f.dgemm_(&ta, &tb, &mm, &nn, &kk, &alpha, a, &la, b, &lb, &beta, c, &lc);
}

pub export fn cblas_cgemm(layout: c_int, transa: c_int, transb: c_int, m: c_int, n: c_int, k: c_int, alpha: *const ComplexF32, a: [*]const ComplexF32, lda: c_int, b: [*]const ComplexF32, ldb: c_int, beta: *const ComplexF32, c: [*]ComplexF32, ldc: c_int) callconv(.c) void {
    if (!validLayout(layout) or !validTrans(transa) or !validTrans(transb)) return;
    var ta = if (layout == CblasRowMajor) transChar(transb) else transChar(transa);
    var tb = if (layout == CblasRowMajor) transChar(transa) else transChar(transb);
    var mm = asBlasInt(if (layout == CblasRowMajor) n else m);
    var nn = asBlasInt(if (layout == CblasRowMajor) m else n);
    var kk = asBlasInt(k);
    var la = asBlasInt(if (layout == CblasRowMajor) ldb else lda);
    var lb = asBlasInt(if (layout == CblasRowMajor) lda else ldb);
    var lc = asBlasInt(ldc);
    if (layout == CblasRowMajor) f.cgemm_(&ta, &tb, &mm, &nn, &kk, alpha, b, &la, a, &lb, beta, c, &lc) else f.cgemm_(&ta, &tb, &mm, &nn, &kk, alpha, a, &la, b, &lb, beta, c, &lc);
}

pub export fn cblas_zgemm(layout: c_int, transa: c_int, transb: c_int, m: c_int, n: c_int, k: c_int, alpha: *const ComplexF64, a: [*]const ComplexF64, lda: c_int, b: [*]const ComplexF64, ldb: c_int, beta: *const ComplexF64, c: [*]ComplexF64, ldc: c_int) callconv(.c) void {
    if (!validLayout(layout) or !validTrans(transa) or !validTrans(transb)) return;
    var ta = if (layout == CblasRowMajor) transChar(transb) else transChar(transa);
    var tb = if (layout == CblasRowMajor) transChar(transa) else transChar(transb);
    var mm = asBlasInt(if (layout == CblasRowMajor) n else m);
    var nn = asBlasInt(if (layout == CblasRowMajor) m else n);
    var kk = asBlasInt(k);
    var la = asBlasInt(if (layout == CblasRowMajor) ldb else lda);
    var lb = asBlasInt(if (layout == CblasRowMajor) lda else ldb);
    var lc = asBlasInt(ldc);
    if (layout == CblasRowMajor) f.zgemm_(&ta, &tb, &mm, &nn, &kk, alpha, b, &la, a, &lb, beta, c, &lc) else f.zgemm_(&ta, &tb, &mm, &nn, &kk, alpha, a, &la, b, &lb, beta, c, &lc);
}

pub export fn cblas_ssymm(layout: c_int, side: c_int, uplo: c_int, m: c_int, n: c_int, alpha: f32, a: [*]const f32, lda: c_int, b: [*]const f32, ldb: c_int, beta: f32, c: [*]f32, ldc: c_int) callconv(.c) void {
    if (!validLayout(layout)) return;
    var ss = if (layout == CblasRowMajor) rowMajorSideChar(side) else sideChar(side);
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var mm = asBlasInt(if (layout == CblasRowMajor) n else m);
    var nn = asBlasInt(if (layout == CblasRowMajor) m else n);
    var la = asBlasInt(lda);
    var lb = asBlasInt(ldb);
    var lc = asBlasInt(ldc);
    f.ssymm_(&ss, &uu, &mm, &nn, &alpha, a, &la, b, &lb, &beta, c, &lc);
}

pub export fn cblas_dsymm(layout: c_int, side: c_int, uplo: c_int, m: c_int, n: c_int, alpha: f64, a: [*]const f64, lda: c_int, b: [*]const f64, ldb: c_int, beta: f64, c: [*]f64, ldc: c_int) callconv(.c) void {
    if (!validLayout(layout)) return;
    var ss = if (layout == CblasRowMajor) rowMajorSideChar(side) else sideChar(side);
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var mm = asBlasInt(if (layout == CblasRowMajor) n else m);
    var nn = asBlasInt(if (layout == CblasRowMajor) m else n);
    var la = asBlasInt(lda);
    var lb = asBlasInt(ldb);
    var lc = asBlasInt(ldc);
    f.dsymm_(&ss, &uu, &mm, &nn, &alpha, a, &la, b, &lb, &beta, c, &lc);
}

pub export fn cblas_csymm(layout: c_int, side: c_int, uplo: c_int, m: c_int, n: c_int, alpha: *const ComplexF32, a: [*]const ComplexF32, lda: c_int, b: [*]const ComplexF32, ldb: c_int, beta: *const ComplexF32, c: [*]ComplexF32, ldc: c_int) callconv(.c) void {
    if (!validLayout(layout)) return;
    var ss = if (layout == CblasRowMajor) rowMajorSideChar(side) else sideChar(side);
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var mm = asBlasInt(if (layout == CblasRowMajor) n else m);
    var nn = asBlasInt(if (layout == CblasRowMajor) m else n);
    var la = asBlasInt(lda);
    var lb = asBlasInt(ldb);
    var lc = asBlasInt(ldc);
    f.csymm_(&ss, &uu, &mm, &nn, alpha, a, &la, b, &lb, beta, c, &lc);
}

pub export fn cblas_zsymm(layout: c_int, side: c_int, uplo: c_int, m: c_int, n: c_int, alpha: *const ComplexF64, a: [*]const ComplexF64, lda: c_int, b: [*]const ComplexF64, ldb: c_int, beta: *const ComplexF64, c: [*]ComplexF64, ldc: c_int) callconv(.c) void {
    if (!validLayout(layout)) return;
    var ss = if (layout == CblasRowMajor) rowMajorSideChar(side) else sideChar(side);
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var mm = asBlasInt(if (layout == CblasRowMajor) n else m);
    var nn = asBlasInt(if (layout == CblasRowMajor) m else n);
    var la = asBlasInt(lda);
    var lb = asBlasInt(ldb);
    var lc = asBlasInt(ldc);
    f.zsymm_(&ss, &uu, &mm, &nn, alpha, a, &la, b, &lb, beta, c, &lc);
}

pub export fn cblas_chemm(layout: c_int, side: c_int, uplo: c_int, m: c_int, n: c_int, alpha: *const ComplexF32, a: [*]const ComplexF32, lda: c_int, b: [*]const ComplexF32, ldb: c_int, beta: *const ComplexF32, c: [*]ComplexF32, ldc: c_int) callconv(.c) void {
    if (!validLayout(layout)) return;
    var ss = if (layout == CblasRowMajor) rowMajorSideChar(side) else sideChar(side);
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var mm = asBlasInt(if (layout == CblasRowMajor) n else m);
    var nn = asBlasInt(if (layout == CblasRowMajor) m else n);
    var la = asBlasInt(lda);
    var lb = asBlasInt(ldb);
    var lc = asBlasInt(ldc);
    f.chemm_(&ss, &uu, &mm, &nn, alpha, a, &la, b, &lb, beta, c, &lc);
}

pub export fn cblas_zhemm(layout: c_int, side: c_int, uplo: c_int, m: c_int, n: c_int, alpha: *const ComplexF64, a: [*]const ComplexF64, lda: c_int, b: [*]const ComplexF64, ldb: c_int, beta: *const ComplexF64, c: [*]ComplexF64, ldc: c_int) callconv(.c) void {
    if (!validLayout(layout)) return;
    var ss = if (layout == CblasRowMajor) rowMajorSideChar(side) else sideChar(side);
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var mm = asBlasInt(if (layout == CblasRowMajor) n else m);
    var nn = asBlasInt(if (layout == CblasRowMajor) m else n);
    var la = asBlasInt(lda);
    var lb = asBlasInt(ldb);
    var lc = asBlasInt(ldc);
    f.zhemm_(&ss, &uu, &mm, &nn, alpha, a, &la, b, &lb, beta, c, &lc);
}

pub export fn cblas_ssyrk(layout: c_int, uplo: c_int, trans: c_int, n: c_int, k: c_int, alpha: f32, a: [*]const f32, lda: c_int, beta: f32, c: [*]f32, ldc: c_int) callconv(.c) void {
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var tt = if (layout == CblasRowMajor) rowMajorTransChar(trans) else transChar(trans);
    var nn = asBlasInt(n);
    var kk = asBlasInt(k);
    var la = asBlasInt(lda);
    var lc = asBlasInt(ldc);
    f.ssyrk_(&uu, &tt, &nn, &kk, &alpha, a, &la, &beta, c, &lc);
}

pub export fn cblas_dsyrk(layout: c_int, uplo: c_int, trans: c_int, n: c_int, k: c_int, alpha: f64, a: [*]const f64, lda: c_int, beta: f64, c: [*]f64, ldc: c_int) callconv(.c) void {
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var tt = if (layout == CblasRowMajor) rowMajorTransChar(trans) else transChar(trans);
    var nn = asBlasInt(n);
    var kk = asBlasInt(k);
    var la = asBlasInt(lda);
    var lc = asBlasInt(ldc);
    f.dsyrk_(&uu, &tt, &nn, &kk, &alpha, a, &la, &beta, c, &lc);
}

pub export fn cblas_csyrk(layout: c_int, uplo: c_int, trans: c_int, n: c_int, k: c_int, alpha: *const ComplexF32, a: [*]const ComplexF32, lda: c_int, beta: *const ComplexF32, c: [*]ComplexF32, ldc: c_int) callconv(.c) void {
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var tt = if (layout == CblasRowMajor) rowMajorTransChar(trans) else transChar(trans);
    var nn = asBlasInt(n);
    var kk = asBlasInt(k);
    var la = asBlasInt(lda);
    var lc = asBlasInt(ldc);
    f.csyrk_(&uu, &tt, &nn, &kk, alpha, a, &la, beta, c, &lc);
}

pub export fn cblas_zsyrk(layout: c_int, uplo: c_int, trans: c_int, n: c_int, k: c_int, alpha: *const ComplexF64, a: [*]const ComplexF64, lda: c_int, beta: *const ComplexF64, c: [*]ComplexF64, ldc: c_int) callconv(.c) void {
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var tt = if (layout == CblasRowMajor) rowMajorTransChar(trans) else transChar(trans);
    var nn = asBlasInt(n);
    var kk = asBlasInt(k);
    var la = asBlasInt(lda);
    var lc = asBlasInt(ldc);
    f.zsyrk_(&uu, &tt, &nn, &kk, alpha, a, &la, beta, c, &lc);
}

pub export fn cblas_cherk(layout: c_int, uplo: c_int, trans: c_int, n: c_int, k: c_int, alpha: f32, a: [*]const ComplexF32, lda: c_int, beta: f32, c: [*]ComplexF32, ldc: c_int) callconv(.c) void {
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var tt = if (layout == CblasRowMajor) rowMajorTransChar(trans) else transChar(trans);
    var nn = asBlasInt(n);
    var kk = asBlasInt(k);
    var la = asBlasInt(lda);
    var lc = asBlasInt(ldc);
    f.cherk_(&uu, &tt, &nn, &kk, &alpha, a, &la, &beta, c, &lc);
}

pub export fn cblas_zherk(layout: c_int, uplo: c_int, trans: c_int, n: c_int, k: c_int, alpha: f64, a: [*]const ComplexF64, lda: c_int, beta: f64, c: [*]ComplexF64, ldc: c_int) callconv(.c) void {
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var tt = if (layout == CblasRowMajor) rowMajorTransChar(trans) else transChar(trans);
    var nn = asBlasInt(n);
    var kk = asBlasInt(k);
    var la = asBlasInt(lda);
    var lc = asBlasInt(ldc);
    f.zherk_(&uu, &tt, &nn, &kk, &alpha, a, &la, &beta, c, &lc);
}

pub export fn cblas_ssyr2k(layout: c_int, uplo: c_int, trans: c_int, n: c_int, k: c_int, alpha: f32, a: [*]const f32, lda: c_int, b: [*]const f32, ldb: c_int, beta: f32, c: [*]f32, ldc: c_int) callconv(.c) void {
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var tt = if (layout == CblasRowMajor) rowMajorTransChar(trans) else transChar(trans);
    var nn = asBlasInt(n);
    var kk = asBlasInt(k);
    var la = asBlasInt(lda);
    var lb = asBlasInt(ldb);
    var lc = asBlasInt(ldc);
    f.ssyr2k_(&uu, &tt, &nn, &kk, &alpha, a, &la, b, &lb, &beta, c, &lc);
}

pub export fn cblas_dsyr2k(layout: c_int, uplo: c_int, trans: c_int, n: c_int, k: c_int, alpha: f64, a: [*]const f64, lda: c_int, b: [*]const f64, ldb: c_int, beta: f64, c: [*]f64, ldc: c_int) callconv(.c) void {
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var tt = if (layout == CblasRowMajor) rowMajorTransChar(trans) else transChar(trans);
    var nn = asBlasInt(n);
    var kk = asBlasInt(k);
    var la = asBlasInt(lda);
    var lb = asBlasInt(ldb);
    var lc = asBlasInt(ldc);
    f.dsyr2k_(&uu, &tt, &nn, &kk, &alpha, a, &la, b, &lb, &beta, c, &lc);
}

pub export fn cblas_csyr2k(layout: c_int, uplo: c_int, trans: c_int, n: c_int, k: c_int, alpha: *const ComplexF32, a: [*]const ComplexF32, lda: c_int, b: [*]const ComplexF32, ldb: c_int, beta: *const ComplexF32, c: [*]ComplexF32, ldc: c_int) callconv(.c) void {
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var tt = if (layout == CblasRowMajor) rowMajorTransChar(trans) else transChar(trans);
    var nn = asBlasInt(n);
    var kk = asBlasInt(k);
    var la = asBlasInt(lda);
    var lb = asBlasInt(ldb);
    var lc = asBlasInt(ldc);
    f.csyr2k_(&uu, &tt, &nn, &kk, alpha, a, &la, b, &lb, beta, c, &lc);
}

pub export fn cblas_zsyr2k(layout: c_int, uplo: c_int, trans: c_int, n: c_int, k: c_int, alpha: *const ComplexF64, a: [*]const ComplexF64, lda: c_int, b: [*]const ComplexF64, ldb: c_int, beta: *const ComplexF64, c: [*]ComplexF64, ldc: c_int) callconv(.c) void {
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var tt = if (layout == CblasRowMajor) rowMajorTransChar(trans) else transChar(trans);
    var nn = asBlasInt(n);
    var kk = asBlasInt(k);
    var la = asBlasInt(lda);
    var lb = asBlasInt(ldb);
    var lc = asBlasInt(ldc);
    f.zsyr2k_(&uu, &tt, &nn, &kk, alpha, a, &la, b, &lb, beta, c, &lc);
}

pub export fn cblas_cher2k(layout: c_int, uplo: c_int, trans: c_int, n: c_int, k: c_int, alpha: *const ComplexF32, a: [*]const ComplexF32, lda: c_int, b: [*]const ComplexF32, ldb: c_int, beta: f32, c: [*]ComplexF32, ldc: c_int) callconv(.c) void {
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var tt = if (layout == CblasRowMajor) rowMajorTransChar(trans) else transChar(trans);
    var nn = asBlasInt(n);
    var kk = asBlasInt(k);
    var la = asBlasInt(lda);
    var lb = asBlasInt(ldb);
    var lc = asBlasInt(ldc);
    f.cher2k_(&uu, &tt, &nn, &kk, alpha, a, &la, b, &lb, &beta, c, &lc);
}

pub export fn cblas_zher2k(layout: c_int, uplo: c_int, trans: c_int, n: c_int, k: c_int, alpha: *const ComplexF64, a: [*]const ComplexF64, lda: c_int, b: [*]const ComplexF64, ldb: c_int, beta: f64, c: [*]ComplexF64, ldc: c_int) callconv(.c) void {
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var tt = if (layout == CblasRowMajor) rowMajorTransChar(trans) else transChar(trans);
    var nn = asBlasInt(n);
    var kk = asBlasInt(k);
    var la = asBlasInt(lda);
    var lb = asBlasInt(ldb);
    var lc = asBlasInt(ldc);
    f.zher2k_(&uu, &tt, &nn, &kk, alpha, a, &la, b, &lb, &beta, c, &lc);
}

pub export fn cblas_strmm(layout: c_int, side: c_int, uplo: c_int, transa: c_int, diag: c_int, m: c_int, n: c_int, alpha: f32, a: [*]const f32, lda: c_int, b: [*]f32, ldb: c_int) callconv(.c) void {
    if (!validLayout(layout) or !validSide(side) or !validUplo(uplo) or !validTrans(transa) or !validDiag(diag)) return;
    var ss = if (layout == CblasRowMajor) rowMajorSideChar(side) else sideChar(side);
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var tt = transChar(transa);
    var dd = diagChar(diag);
    var mm = asBlasInt(if (layout == CblasRowMajor) n else m);
    var nn = asBlasInt(if (layout == CblasRowMajor) m else n);
    var la = asBlasInt(lda);
    var lb = asBlasInt(ldb);
    f.strmm_(&ss, &uu, &tt, &dd, &mm, &nn, &alpha, a, &la, b, &lb);
}

pub export fn cblas_dtrmm(layout: c_int, side: c_int, uplo: c_int, transa: c_int, diag: c_int, m: c_int, n: c_int, alpha: f64, a: [*]const f64, lda: c_int, b: [*]f64, ldb: c_int) callconv(.c) void {
    if (!validLayout(layout) or !validSide(side) or !validUplo(uplo) or !validTrans(transa) or !validDiag(diag)) return;
    var ss = if (layout == CblasRowMajor) rowMajorSideChar(side) else sideChar(side);
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var tt = transChar(transa);
    var dd = diagChar(diag);
    var mm = asBlasInt(if (layout == CblasRowMajor) n else m);
    var nn = asBlasInt(if (layout == CblasRowMajor) m else n);
    var la = asBlasInt(lda);
    var lb = asBlasInt(ldb);
    f.dtrmm_(&ss, &uu, &tt, &dd, &mm, &nn, &alpha, a, &la, b, &lb);
}

pub export fn cblas_ctrmm(layout: c_int, side: c_int, uplo: c_int, transa: c_int, diag: c_int, m: c_int, n: c_int, alpha: *const ComplexF32, a: [*]const ComplexF32, lda: c_int, b: [*]ComplexF32, ldb: c_int) callconv(.c) void {
    if (!validLayout(layout) or !validSide(side) or !validUplo(uplo) or !validTrans(transa) or !validDiag(diag)) return;
    var ss = if (layout == CblasRowMajor) rowMajorSideChar(side) else sideChar(side);
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var tt = transChar(transa);
    var dd = diagChar(diag);
    var mm = asBlasInt(if (layout == CblasRowMajor) n else m);
    var nn = asBlasInt(if (layout == CblasRowMajor) m else n);
    var la = asBlasInt(lda);
    var lb = asBlasInt(ldb);
    f.ctrmm_(&ss, &uu, &tt, &dd, &mm, &nn, alpha, a, &la, b, &lb);
}

pub export fn cblas_ztrmm(layout: c_int, side: c_int, uplo: c_int, transa: c_int, diag: c_int, m: c_int, n: c_int, alpha: *const ComplexF64, a: [*]const ComplexF64, lda: c_int, b: [*]ComplexF64, ldb: c_int) callconv(.c) void {
    if (!validLayout(layout) or !validSide(side) or !validUplo(uplo) or !validTrans(transa) or !validDiag(diag)) return;
    var ss = if (layout == CblasRowMajor) rowMajorSideChar(side) else sideChar(side);
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var tt = transChar(transa);
    var dd = diagChar(diag);
    var mm = asBlasInt(if (layout == CblasRowMajor) n else m);
    var nn = asBlasInt(if (layout == CblasRowMajor) m else n);
    var la = asBlasInt(lda);
    var lb = asBlasInt(ldb);
    f.ztrmm_(&ss, &uu, &tt, &dd, &mm, &nn, alpha, a, &la, b, &lb);
}

pub export fn cblas_strsm(layout: c_int, side: c_int, uplo: c_int, transa: c_int, diag: c_int, m: c_int, n: c_int, alpha: f32, a: [*]const f32, lda: c_int, b: [*]f32, ldb: c_int) callconv(.c) void {
    if (!validLayout(layout) or !validSide(side) or !validUplo(uplo) or !validTrans(transa) or !validDiag(diag)) return;
    var ss = if (layout == CblasRowMajor) rowMajorSideChar(side) else sideChar(side);
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var tt = transChar(transa);
    var dd = diagChar(diag);
    var mm = asBlasInt(if (layout == CblasRowMajor) n else m);
    var nn = asBlasInt(if (layout == CblasRowMajor) m else n);
    var la = asBlasInt(lda);
    var lb = asBlasInt(ldb);
    f.strsm_(&ss, &uu, &tt, &dd, &mm, &nn, &alpha, a, &la, b, &lb);
}

pub export fn cblas_dtrsm(layout: c_int, side: c_int, uplo: c_int, transa: c_int, diag: c_int, m: c_int, n: c_int, alpha: f64, a: [*]const f64, lda: c_int, b: [*]f64, ldb: c_int) callconv(.c) void {
    if (!validLayout(layout) or !validSide(side) or !validUplo(uplo) or !validTrans(transa) or !validDiag(diag)) return;
    var ss = if (layout == CblasRowMajor) rowMajorSideChar(side) else sideChar(side);
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var tt = transChar(transa);
    var dd = diagChar(diag);
    var mm = asBlasInt(if (layout == CblasRowMajor) n else m);
    var nn = asBlasInt(if (layout == CblasRowMajor) m else n);
    var la = asBlasInt(lda);
    var lb = asBlasInt(ldb);
    f.dtrsm_(&ss, &uu, &tt, &dd, &mm, &nn, &alpha, a, &la, b, &lb);
}

pub export fn cblas_ctrsm(layout: c_int, side: c_int, uplo: c_int, transa: c_int, diag: c_int, m: c_int, n: c_int, alpha: *const ComplexF32, a: [*]const ComplexF32, lda: c_int, b: [*]ComplexF32, ldb: c_int) callconv(.c) void {
    if (!validLayout(layout) or !validSide(side) or !validUplo(uplo) or !validTrans(transa) or !validDiag(diag)) return;
    var ss = if (layout == CblasRowMajor) rowMajorSideChar(side) else sideChar(side);
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var tt = transChar(transa);
    var dd = diagChar(diag);
    var mm = asBlasInt(if (layout == CblasRowMajor) n else m);
    var nn = asBlasInt(if (layout == CblasRowMajor) m else n);
    var la = asBlasInt(lda);
    var lb = asBlasInt(ldb);
    f.ctrsm_(&ss, &uu, &tt, &dd, &mm, &nn, alpha, a, &la, b, &lb);
}

pub export fn cblas_ztrsm(layout: c_int, side: c_int, uplo: c_int, transa: c_int, diag: c_int, m: c_int, n: c_int, alpha: *const ComplexF64, a: [*]const ComplexF64, lda: c_int, b: [*]ComplexF64, ldb: c_int) callconv(.c) void {
    if (!validLayout(layout) or !validSide(side) or !validUplo(uplo) or !validTrans(transa) or !validDiag(diag)) return;
    var ss = if (layout == CblasRowMajor) rowMajorSideChar(side) else sideChar(side);
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var tt = transChar(transa);
    var dd = diagChar(diag);
    var mm = asBlasInt(if (layout == CblasRowMajor) n else m);
    var nn = asBlasInt(if (layout == CblasRowMajor) m else n);
    var la = asBlasInt(lda);
    var lb = asBlasInt(ldb);
    f.ztrsm_(&ss, &uu, &tt, &dd, &mm, &nn, alpha, a, &la, b, &lb);
}
