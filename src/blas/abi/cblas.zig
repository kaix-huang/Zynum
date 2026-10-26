// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const types = @import("zynum-core").blas.types;
const core = @import("zynum-core").internal.blas_unchecked;
const execution = @import("zynum-core").internal.numerics.Execution;
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
    var operation = execution.beginDefault();
    defer operation.end();
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.internal_admitted.sswap_(&operation, &nn, x, &ix, y, &iy);
}
pub export fn cblas_dswap(n: c_int, x: [*]f64, incx: c_int, y: [*]f64, incy: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.internal_admitted.dswap_(&operation, &nn, x, &ix, y, &iy);
}
pub export fn cblas_cswap(n: c_int, x: [*]ComplexF32, incx: c_int, y: [*]ComplexF32, incy: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.internal_admitted.cswap_(&operation, &nn, x, &ix, y, &iy);
}
pub export fn cblas_zswap(n: c_int, x: [*]ComplexF64, incx: c_int, y: [*]ComplexF64, incy: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.internal_admitted.zswap_(&operation, &nn, x, &ix, y, &iy);
}

pub export fn cblas_scopy(n: c_int, x: [*]const f32, incx: c_int, y: [*]f32, incy: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.internal_admitted.scopy_(&operation, &nn, x, &ix, y, &iy);
}
pub export fn cblas_dcopy(n: c_int, x: [*]const f64, incx: c_int, y: [*]f64, incy: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.internal_admitted.dcopy_(&operation, &nn, x, &ix, y, &iy);
}
pub export fn cblas_ccopy(n: c_int, x: [*]const ComplexF32, incx: c_int, y: [*]ComplexF32, incy: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.internal_admitted.ccopy_(&operation, &nn, x, &ix, y, &iy);
}
pub export fn cblas_zcopy(n: c_int, x: [*]const ComplexF64, incx: c_int, y: [*]ComplexF64, incy: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.internal_admitted.zcopy_(&operation, &nn, x, &ix, y, &iy);
}

pub export fn cblas_saxpy(n: c_int, alpha: f32, x: [*]const f32, incx: c_int, y: [*]f32, incy: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.internal_admitted.saxpy_(&operation, &nn, &alpha, x, &ix, y, &iy);
}
pub export fn cblas_daxpy(n: c_int, alpha: f64, x: [*]const f64, incx: c_int, y: [*]f64, incy: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.internal_admitted.daxpy_(&operation, &nn, &alpha, x, &ix, y, &iy);
}
pub export fn cblas_caxpy(n: c_int, alpha: *const ComplexF32, x: [*]const ComplexF32, incx: c_int, y: [*]ComplexF32, incy: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.internal_admitted.caxpy_(&operation, &nn, alpha, x, &ix, y, &iy);
}
pub export fn cblas_zaxpy(n: c_int, alpha: *const ComplexF64, x: [*]const ComplexF64, incx: c_int, y: [*]ComplexF64, incy: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.internal_admitted.zaxpy_(&operation, &nn, alpha, x, &ix, y, &iy);
}

pub export fn cblas_saxpby(n: c_int, alpha: f32, x: [*]const f32, incx: c_int, beta: f32, y: [*]f32, incy: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.internal_admitted.saxpby_(&operation, &nn, &alpha, x, &ix, &beta, y, &iy);
}
pub export fn cblas_daxpby(n: c_int, alpha: f64, x: [*]const f64, incx: c_int, beta: f64, y: [*]f64, incy: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.internal_admitted.daxpby_(&operation, &nn, &alpha, x, &ix, &beta, y, &iy);
}
pub export fn cblas_caxpby(n: c_int, alpha: *const ComplexF32, x: [*]const ComplexF32, incx: c_int, beta: *const ComplexF32, y: [*]ComplexF32, incy: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.internal_admitted.caxpby_(&operation, &nn, alpha, x, &ix, beta, y, &iy);
}
pub export fn cblas_zaxpby(n: c_int, alpha: *const ComplexF64, x: [*]const ComplexF64, incx: c_int, beta: *const ComplexF64, y: [*]ComplexF64, incy: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.internal_admitted.zaxpby_(&operation, &nn, alpha, x, &ix, beta, y, &iy);
}

pub export fn cblas_sscal(n: c_int, alpha: f32, x: [*]f32, incx: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    f.internal_admitted.sscal_(&operation, &nn, &alpha, x, &ix);
}
pub export fn cblas_dscal(n: c_int, alpha: f64, x: [*]f64, incx: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    f.internal_admitted.dscal_(&operation, &nn, &alpha, x, &ix);
}
pub export fn cblas_cscal(n: c_int, alpha: *const ComplexF32, x: [*]ComplexF32, incx: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    f.internal_admitted.cscal_(&operation, &nn, alpha, x, &ix);
}
pub export fn cblas_zscal(n: c_int, alpha: *const ComplexF64, x: [*]ComplexF64, incx: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    f.internal_admitted.zscal_(&operation, &nn, alpha, x, &ix);
}
pub export fn cblas_csscal(n: c_int, alpha: f32, x: [*]ComplexF32, incx: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    f.internal_admitted.csscal_(&operation, &nn, &alpha, x, &ix);
}
pub export fn cblas_zdscal(n: c_int, alpha: f64, x: [*]ComplexF64, incx: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    f.internal_admitted.zdscal_(&operation, &nn, &alpha, x, &ix);
}

pub export fn cblas_sdot(n: c_int, x: [*]const f32, incx: c_int, y: [*]const f32, incy: c_int) callconv(.c) f32 {
    var operation = execution.beginDefault();
    defer operation.end();
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    return f.internal_admitted.sdot_(&operation, &nn, x, &ix, y, &iy);
}
pub export fn cblas_ddot(n: c_int, x: [*]const f64, incx: c_int, y: [*]const f64, incy: c_int) callconv(.c) f64 {
    var operation = execution.beginDefault();
    defer operation.end();
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    return f.internal_admitted.ddot_(&operation, &nn, x, &ix, y, &iy);
}
pub export fn cblas_cdotu_sub(n: c_int, x: [*]const ComplexF32, incx: c_int, y: [*]const ComplexF32, incy: c_int, out: *ComplexF32) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.internal_admitted.cdotu_sub_(&operation, &nn, x, &ix, y, &iy, out);
}
pub export fn cblas_zdotu_sub(n: c_int, x: [*]const ComplexF64, incx: c_int, y: [*]const ComplexF64, incy: c_int, out: *ComplexF64) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.internal_admitted.zdotu_sub_(&operation, &nn, x, &ix, y, &iy, out);
}
pub export fn cblas_cdotc_sub(n: c_int, x: [*]const ComplexF32, incx: c_int, y: [*]const ComplexF32, incy: c_int, out: *ComplexF32) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.internal_admitted.cdotc_sub_(&operation, &nn, x, &ix, y, &iy, out);
}
pub export fn cblas_zdotc_sub(n: c_int, x: [*]const ComplexF64, incx: c_int, y: [*]const ComplexF64, incy: c_int, out: *ComplexF64) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.internal_admitted.zdotc_sub_(&operation, &nn, x, &ix, y, &iy, out);
}

pub export fn cblas_snrm2(n: c_int, x: [*]const f32, incx: c_int) callconv(.c) f32 {
    var operation = execution.beginDefault();
    defer operation.end();
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    return f.internal_admitted.snrm2_(&operation, &nn, x, &ix);
}
pub export fn cblas_dnrm2(n: c_int, x: [*]const f64, incx: c_int) callconv(.c) f64 {
    var operation = execution.beginDefault();
    defer operation.end();
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    return f.internal_admitted.dnrm2_(&operation, &nn, x, &ix);
}
pub export fn cblas_scnrm2(n: c_int, x: [*]const ComplexF32, incx: c_int) callconv(.c) f32 {
    var operation = execution.beginDefault();
    defer operation.end();
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    return f.internal_admitted.scnrm2_(&operation, &nn, x, &ix);
}
pub export fn cblas_dznrm2(n: c_int, x: [*]const ComplexF64, incx: c_int) callconv(.c) f64 {
    var operation = execution.beginDefault();
    defer operation.end();
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    return f.internal_admitted.dznrm2_(&operation, &nn, x, &ix);
}

pub export fn cblas_sasum(n: c_int, x: [*]const f32, incx: c_int) callconv(.c) f32 {
    var operation = execution.beginDefault();
    defer operation.end();
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    return f.internal_admitted.sasum_(&operation, &nn, x, &ix);
}
pub export fn cblas_dasum(n: c_int, x: [*]const f64, incx: c_int) callconv(.c) f64 {
    var operation = execution.beginDefault();
    defer operation.end();
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    return f.internal_admitted.dasum_(&operation, &nn, x, &ix);
}
pub export fn cblas_scasum(n: c_int, x: [*]const ComplexF32, incx: c_int) callconv(.c) f32 {
    var operation = execution.beginDefault();
    defer operation.end();
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    return f.internal_admitted.scasum_(&operation, &nn, x, &ix);
}
pub export fn cblas_dzasum(n: c_int, x: [*]const ComplexF64, incx: c_int) callconv(.c) f64 {
    var operation = execution.beginDefault();
    defer operation.end();
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    return f.internal_admitted.dzasum_(&operation, &nn, x, &ix);
}

pub export fn cblas_isamax(n: c_int, x: [*]const f32, incx: c_int) callconv(.c) c_int {
    var operation = execution.beginDefault();
    defer operation.end();
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    return cblasIndex(f.internal_admitted.isamax_(&operation, &nn, x, &ix));
}
pub export fn cblas_idamax(n: c_int, x: [*]const f64, incx: c_int) callconv(.c) c_int {
    var operation = execution.beginDefault();
    defer operation.end();
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    return cblasIndex(f.internal_admitted.idamax_(&operation, &nn, x, &ix));
}
pub export fn cblas_icamax(n: c_int, x: [*]const ComplexF32, incx: c_int) callconv(.c) c_int {
    var operation = execution.beginDefault();
    defer operation.end();
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    return cblasIndex(f.internal_admitted.icamax_(&operation, &nn, x, &ix));
}
pub export fn cblas_izamax(n: c_int, x: [*]const ComplexF64, incx: c_int) callconv(.c) c_int {
    var operation = execution.beginDefault();
    defer operation.end();
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    return cblasIndex(f.internal_admitted.izamax_(&operation, &nn, x, &ix));
}

pub export fn cblas_sdsdot(n: c_int, sb: f32, x: [*]const f32, incx: c_int, y: [*]const f32, incy: c_int) callconv(.c) f32 {
    var operation = execution.beginDefault();
    defer operation.end();
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    return f.internal_admitted.sdsdot_(&operation, &nn, &sb, x, &ix, y, &iy);
}
pub export fn cblas_dsdot(n: c_int, x: [*]const f32, incx: c_int, y: [*]const f32, incy: c_int) callconv(.c) f64 {
    var operation = execution.beginDefault();
    defer operation.end();
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    return f.internal_admitted.dsdot_(&operation, &nn, x, &ix, y, &iy);
}

pub export fn cblas_srotg(a: *f32, b: *f32, c: *f32, s: *f32) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    f.internal_admitted.srotg_(&operation, a, b, c, s);
}
pub export fn cblas_drotg(a: *f64, b: *f64, c: *f64, s: *f64) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    f.internal_admitted.drotg_(&operation, a, b, c, s);
}
pub export fn cblas_crotg(a: *ComplexF32, b: *ComplexF32, c: *f32, s: *ComplexF32) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    f.internal_admitted.crotg_(&operation, a, b, c, s);
}
pub export fn cblas_zrotg(a: *ComplexF64, b: *ComplexF64, c: *f64, s: *ComplexF64) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    f.internal_admitted.zrotg_(&operation, a, b, c, s);
}

pub export fn cblas_srot(n: c_int, x: [*]f32, incx: c_int, y: [*]f32, incy: c_int, c: f32, s: f32) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.internal_admitted.srot_(&operation, &nn, x, &ix, y, &iy, &c, &s);
}
pub export fn cblas_drot(n: c_int, x: [*]f64, incx: c_int, y: [*]f64, incy: c_int, c: f64, s: f64) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.internal_admitted.drot_(&operation, &nn, x, &ix, y, &iy, &c, &s);
}
pub export fn cblas_csrot(n: c_int, x: [*]ComplexF32, incx: c_int, y: [*]ComplexF32, incy: c_int, c: f32, s: f32) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.internal_admitted.csrot_(&operation, &nn, x, &ix, y, &iy, &c, &s);
}
pub export fn cblas_zdrot(n: c_int, x: [*]ComplexF64, incx: c_int, y: [*]ComplexF64, incy: c_int, c: f64, s: f64) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.internal_admitted.zdrot_(&operation, &nn, x, &ix, y, &iy, &c, &s);
}

pub export fn cblas_srotm(n: c_int, x: [*]f32, incx: c_int, y: [*]f32, incy: c_int, param: [*]const f32) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.internal_admitted.srotm_(&operation, &nn, x, &ix, y, &iy, param);
}
pub export fn cblas_drotm(n: c_int, x: [*]f64, incx: c_int, y: [*]f64, incy: c_int, param: [*]const f64) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.internal_admitted.drotm_(&operation, &nn, x, &ix, y, &iy, param);
}
pub export fn cblas_srotmg(d1: *f32, d2: *f32, x1: *f32, y1: f32, param: [*]f32) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    f.internal_admitted.srotmg_(&operation, d1, d2, x1, &y1, param);
}
pub export fn cblas_drotmg(d1: *f64, d2: *f64, x1: *f64, y1: f64, param: [*]f64) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    f.internal_admitted.drotmg_(&operation, d1, d2, x1, &y1, param);
}

// Level 2 CBLAS wrappers.
pub export fn cblas_sgemv(layout: c_int, trans: c_int, m: c_int, n: c_int, alpha: f32, a: [*]const f32, lda: c_int, x: [*]const f32, incx: c_int, beta: f32, y: [*]f32, incy: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    if (!validLayout(layout)) return;
    var mm = asBlasInt(if (layout == CblasRowMajor) n else m);
    var nn = asBlasInt(if (layout == CblasRowMajor) m else n);
    var tt = if (layout == CblasRowMajor) rowMajorGemvTrans(trans) else transChar(trans);
    var aa = asBlasInt(lda);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.internal_admitted.sgemv_(&operation, &tt, &mm, &nn, &alpha, a, &aa, x, &ix, &beta, y, &iy);
}

pub export fn cblas_dgemv(layout: c_int, trans: c_int, m: c_int, n: c_int, alpha: f64, a: [*]const f64, lda: c_int, x: [*]const f64, incx: c_int, beta: f64, y: [*]f64, incy: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    if (!validLayout(layout)) return;
    var mm = asBlasInt(if (layout == CblasRowMajor) n else m);
    var nn = asBlasInt(if (layout == CblasRowMajor) m else n);
    var tt = if (layout == CblasRowMajor) rowMajorGemvTrans(trans) else transChar(trans);
    var aa = asBlasInt(lda);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.internal_admitted.dgemv_(&operation, &tt, &mm, &nn, &alpha, a, &aa, x, &ix, &beta, y, &iy);
}

pub export fn cblas_cgemv(layout: c_int, trans: c_int, m: c_int, n: c_int, alpha: *const ComplexF32, a: [*]const ComplexF32, lda: c_int, x: [*]const ComplexF32, incx: c_int, beta: *const ComplexF32, y: [*]ComplexF32, incy: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
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
    f.internal_admitted.cgemv_(&operation, &tt, &mm, &nn, alpha, a, &aa, x, &ix, beta, y, &iy);
}

pub export fn cblas_zgemv(layout: c_int, trans: c_int, m: c_int, n: c_int, alpha: *const ComplexF64, a: [*]const ComplexF64, lda: c_int, x: [*]const ComplexF64, incx: c_int, beta: *const ComplexF64, y: [*]ComplexF64, incy: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
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
    f.internal_admitted.zgemv_(&operation, &tt, &mm, &nn, alpha, a, &aa, x, &ix, beta, y, &iy);
}

pub export fn cblas_sgbmv(layout: c_int, trans: c_int, m: c_int, n: c_int, kl: c_int, ku: c_int, alpha: f32, a: [*]const f32, lda: c_int, x: [*]const f32, incx: c_int, beta: f32, y: [*]f32, incy: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    if (!validLayout(layout)) return;
    var mm = asBlasInt(if (layout == CblasRowMajor) n else m);
    var nn = asBlasInt(if (layout == CblasRowMajor) m else n);
    var kll = asBlasInt(if (layout == CblasRowMajor) ku else kl);
    var kuu = asBlasInt(if (layout == CblasRowMajor) kl else ku);
    var tt = if (layout == CblasRowMajor) rowMajorGemvTrans(trans) else transChar(trans);
    var aa = asBlasInt(lda);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.internal_admitted.sgbmv_(&operation, &tt, &mm, &nn, &kll, &kuu, &alpha, a, &aa, x, &ix, &beta, y, &iy);
}

pub export fn cblas_dgbmv(layout: c_int, trans: c_int, m: c_int, n: c_int, kl: c_int, ku: c_int, alpha: f64, a: [*]const f64, lda: c_int, x: [*]const f64, incx: c_int, beta: f64, y: [*]f64, incy: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    if (!validLayout(layout)) return;
    var mm = asBlasInt(if (layout == CblasRowMajor) n else m);
    var nn = asBlasInt(if (layout == CblasRowMajor) m else n);
    var kll = asBlasInt(if (layout == CblasRowMajor) ku else kl);
    var kuu = asBlasInt(if (layout == CblasRowMajor) kl else ku);
    var tt = if (layout == CblasRowMajor) rowMajorGemvTrans(trans) else transChar(trans);
    var aa = asBlasInt(lda);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.internal_admitted.dgbmv_(&operation, &tt, &mm, &nn, &kll, &kuu, &alpha, a, &aa, x, &ix, &beta, y, &iy);
}

pub export fn cblas_cgbmv(layout: c_int, trans: c_int, m: c_int, n: c_int, kl: c_int, ku: c_int, alpha: *const ComplexF32, a: [*]const ComplexF32, lda: c_int, x: [*]const ComplexF32, incx: c_int, beta: *const ComplexF32, y: [*]ComplexF32, incy: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
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
    f.internal_admitted.cgbmv_(&operation, &tt, &mm, &nn, &kll, &kuu, alpha, a, &aa, x, &ix, beta, y, &iy);
}

pub export fn cblas_zgbmv(layout: c_int, trans: c_int, m: c_int, n: c_int, kl: c_int, ku: c_int, alpha: *const ComplexF64, a: [*]const ComplexF64, lda: c_int, x: [*]const ComplexF64, incx: c_int, beta: *const ComplexF64, y: [*]ComplexF64, incy: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
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
    f.internal_admitted.zgbmv_(&operation, &tt, &mm, &nn, &kll, &kuu, alpha, a, &aa, x, &ix, beta, y, &iy);
}

pub export fn cblas_ssymv(layout: c_int, uplo: c_int, n: c_int, alpha: f32, a: [*]const f32, lda: c_int, x: [*]const f32, incx: c_int, beta: f32, y: [*]f32, incy: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var nn = asBlasInt(n);
    var aa = asBlasInt(lda);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.internal_admitted.ssymv_(&operation, &uu, &nn, &alpha, a, &aa, x, &ix, &beta, y, &iy);
}
pub export fn cblas_dsymv(layout: c_int, uplo: c_int, n: c_int, alpha: f64, a: [*]const f64, lda: c_int, x: [*]const f64, incx: c_int, beta: f64, y: [*]f64, incy: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var nn = asBlasInt(n);
    var aa = asBlasInt(lda);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.internal_admitted.dsymv_(&operation, &uu, &nn, &alpha, a, &aa, x, &ix, &beta, y, &iy);
}
pub export fn cblas_chemv(layout: c_int, uplo: c_int, n: c_int, alpha: *const ComplexF32, a: [*]const ComplexF32, lda: c_int, x: [*]const ComplexF32, incx: c_int, beta: *const ComplexF32, y: [*]ComplexF32, incy: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var nn = asBlasInt(n);
    var aa = asBlasInt(lda);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    if (layout == CblasRowMajor)
        f.internal_admitted.hermitianMvView(ComplexF32, .dense, &operation, &uu, nn, 0, alpha, a, aa, x, ix, beta, y, iy, true)
    else
        f.internal_admitted.chemv_(&operation, &uu, &nn, alpha, a, &aa, x, &ix, beta, y, &iy);
}
pub export fn cblas_zhemv(layout: c_int, uplo: c_int, n: c_int, alpha: *const ComplexF64, a: [*]const ComplexF64, lda: c_int, x: [*]const ComplexF64, incx: c_int, beta: *const ComplexF64, y: [*]ComplexF64, incy: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var nn = asBlasInt(n);
    var aa = asBlasInt(lda);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    if (layout == CblasRowMajor)
        f.internal_admitted.hermitianMvView(ComplexF64, .dense, &operation, &uu, nn, 0, alpha, a, aa, x, ix, beta, y, iy, true)
    else
        f.internal_admitted.zhemv_(&operation, &uu, &nn, alpha, a, &aa, x, &ix, beta, y, &iy);
}

pub export fn cblas_ssbmv(layout: c_int, uplo: c_int, n: c_int, k: c_int, alpha: f32, a: [*]const f32, lda: c_int, x: [*]const f32, incx: c_int, beta: f32, y: [*]f32, incy: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var nn = asBlasInt(n);
    var kk = asBlasInt(k);
    var aa = asBlasInt(lda);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.internal_admitted.ssbmv_(&operation, &uu, &nn, &kk, &alpha, a, &aa, x, &ix, &beta, y, &iy);
}
pub export fn cblas_dsbmv(layout: c_int, uplo: c_int, n: c_int, k: c_int, alpha: f64, a: [*]const f64, lda: c_int, x: [*]const f64, incx: c_int, beta: f64, y: [*]f64, incy: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var nn = asBlasInt(n);
    var kk = asBlasInt(k);
    var aa = asBlasInt(lda);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.internal_admitted.dsbmv_(&operation, &uu, &nn, &kk, &alpha, a, &aa, x, &ix, &beta, y, &iy);
}
pub export fn cblas_chbmv(layout: c_int, uplo: c_int, n: c_int, k: c_int, alpha: *const ComplexF32, a: [*]const ComplexF32, lda: c_int, x: [*]const ComplexF32, incx: c_int, beta: *const ComplexF32, y: [*]ComplexF32, incy: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var nn = asBlasInt(n);
    var kk = asBlasInt(k);
    var aa = asBlasInt(lda);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    if (layout == CblasRowMajor)
        f.internal_admitted.hermitianMvView(ComplexF32, .band, &operation, &uu, nn, kk, alpha, a, aa, x, ix, beta, y, iy, true)
    else
        f.internal_admitted.chbmv_(&operation, &uu, &nn, &kk, alpha, a, &aa, x, &ix, beta, y, &iy);
}
pub export fn cblas_zhbmv(layout: c_int, uplo: c_int, n: c_int, k: c_int, alpha: *const ComplexF64, a: [*]const ComplexF64, lda: c_int, x: [*]const ComplexF64, incx: c_int, beta: *const ComplexF64, y: [*]ComplexF64, incy: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var nn = asBlasInt(n);
    var kk = asBlasInt(k);
    var aa = asBlasInt(lda);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    if (layout == CblasRowMajor)
        f.internal_admitted.hermitianMvView(ComplexF64, .band, &operation, &uu, nn, kk, alpha, a, aa, x, ix, beta, y, iy, true)
    else
        f.internal_admitted.zhbmv_(&operation, &uu, &nn, &kk, alpha, a, &aa, x, &ix, beta, y, &iy);
}

pub export fn cblas_sspmv(layout: c_int, uplo: c_int, n: c_int, alpha: f32, ap: [*]const f32, x: [*]const f32, incx: c_int, beta: f32, y: [*]f32, incy: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.internal_admitted.sspmv_(&operation, &uu, &nn, &alpha, ap, x, &ix, &beta, y, &iy);
}
pub export fn cblas_dspmv(layout: c_int, uplo: c_int, n: c_int, alpha: f64, ap: [*]const f64, x: [*]const f64, incx: c_int, beta: f64, y: [*]f64, incy: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.internal_admitted.dspmv_(&operation, &uu, &nn, &alpha, ap, x, &ix, &beta, y, &iy);
}
pub export fn cblas_chpmv(layout: c_int, uplo: c_int, n: c_int, alpha: *const ComplexF32, ap: [*]const ComplexF32, x: [*]const ComplexF32, incx: c_int, beta: *const ComplexF32, y: [*]ComplexF32, incy: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    if (layout == CblasRowMajor)
        f.internal_admitted.hermitianMvView(ComplexF32, .packed_storage, &operation, &uu, nn, 0, alpha, ap, 0, x, ix, beta, y, iy, true)
    else
        f.internal_admitted.chpmv_(&operation, &uu, &nn, alpha, ap, x, &ix, beta, y, &iy);
}
pub export fn cblas_zhpmv(layout: c_int, uplo: c_int, n: c_int, alpha: *const ComplexF64, ap: [*]const ComplexF64, x: [*]const ComplexF64, incx: c_int, beta: *const ComplexF64, y: [*]ComplexF64, incy: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    if (layout == CblasRowMajor)
        f.internal_admitted.hermitianMvView(ComplexF64, .packed_storage, &operation, &uu, nn, 0, alpha, ap, 0, x, ix, beta, y, iy, true)
    else
        f.internal_admitted.zhpmv_(&operation, &uu, &nn, alpha, ap, x, &ix, beta, y, &iy);
}

pub export fn cblas_strmv(layout: c_int, uplo: c_int, trans: c_int, diag: c_int, n: c_int, a: [*]const f32, lda: c_int, x: [*]f32, incx: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var tt = if (layout == CblasRowMajor) rowMajorTransChar(if (trans == CblasConjTrans) CblasTrans else trans) else transChar(trans);
    var dd = diagChar(diag);
    var nn = asBlasInt(n);
    var aa = asBlasInt(lda);
    var ix = asBlasInt(incx);
    f.internal_admitted.strmv_(&operation, &uu, &tt, &dd, &nn, a, &aa, x, &ix);
}
pub export fn cblas_dtrmv(layout: c_int, uplo: c_int, trans: c_int, diag: c_int, n: c_int, a: [*]const f64, lda: c_int, x: [*]f64, incx: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var tt = if (layout == CblasRowMajor) rowMajorTransChar(if (trans == CblasConjTrans) CblasTrans else trans) else transChar(trans);
    var dd = diagChar(diag);
    var nn = asBlasInt(n);
    var aa = asBlasInt(lda);
    var ix = asBlasInt(incx);
    f.internal_admitted.dtrmv_(&operation, &uu, &tt, &dd, &nn, a, &aa, x, &ix);
}
pub export fn cblas_ctrmv(layout: c_int, uplo: c_int, trans: c_int, diag: c_int, n: c_int, a: [*]const ComplexF32, lda: c_int, x: [*]ComplexF32, incx: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
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
    f.internal_admitted.ctrmv_(&operation, &uu, &tt, &dd, &nn, a, &aa, x, &ix);
}
pub export fn cblas_ztrmv(layout: c_int, uplo: c_int, trans: c_int, diag: c_int, n: c_int, a: [*]const ComplexF64, lda: c_int, x: [*]ComplexF64, incx: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
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
    f.internal_admitted.ztrmv_(&operation, &uu, &tt, &dd, &nn, a, &aa, x, &ix);
}

pub export fn cblas_stbmv(layout: c_int, uplo: c_int, trans: c_int, diag: c_int, n: c_int, k: c_int, a: [*]const f32, lda: c_int, x: [*]f32, incx: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var tt = if (layout == CblasRowMajor) rowMajorTransChar(if (trans == CblasConjTrans) CblasTrans else trans) else transChar(trans);
    var dd = diagChar(diag);
    var nn = asBlasInt(n);
    var kk = asBlasInt(k);
    var aa = asBlasInt(lda);
    var ix = asBlasInt(incx);
    f.internal_admitted.stbmv_(&operation, &uu, &tt, &dd, &nn, &kk, a, &aa, x, &ix);
}
pub export fn cblas_dtbmv(layout: c_int, uplo: c_int, trans: c_int, diag: c_int, n: c_int, k: c_int, a: [*]const f64, lda: c_int, x: [*]f64, incx: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var tt = if (layout == CblasRowMajor) rowMajorTransChar(if (trans == CblasConjTrans) CblasTrans else trans) else transChar(trans);
    var dd = diagChar(diag);
    var nn = asBlasInt(n);
    var kk = asBlasInt(k);
    var aa = asBlasInt(lda);
    var ix = asBlasInt(incx);
    f.internal_admitted.dtbmv_(&operation, &uu, &tt, &dd, &nn, &kk, a, &aa, x, &ix);
}
pub export fn cblas_ctbmv(layout: c_int, uplo: c_int, trans: c_int, diag: c_int, n: c_int, k: c_int, a: [*]const ComplexF32, lda: c_int, x: [*]ComplexF32, incx: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
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
    f.internal_admitted.ctbmv_(&operation, &uu, &tt, &dd, &nn, &kk, a, &aa, x, &ix);
}
pub export fn cblas_ztbmv(layout: c_int, uplo: c_int, trans: c_int, diag: c_int, n: c_int, k: c_int, a: [*]const ComplexF64, lda: c_int, x: [*]ComplexF64, incx: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
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
    f.internal_admitted.ztbmv_(&operation, &uu, &tt, &dd, &nn, &kk, a, &aa, x, &ix);
}

pub export fn cblas_stpmv(layout: c_int, uplo: c_int, trans: c_int, diag: c_int, n: c_int, ap: [*]const f32, x: [*]f32, incx: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var tt = if (layout == CblasRowMajor) rowMajorTransChar(if (trans == CblasConjTrans) CblasTrans else trans) else transChar(trans);
    var dd = diagChar(diag);
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    f.internal_admitted.stpmv_(&operation, &uu, &tt, &dd, &nn, ap, x, &ix);
}
pub export fn cblas_dtpmv(layout: c_int, uplo: c_int, trans: c_int, diag: c_int, n: c_int, ap: [*]const f64, x: [*]f64, incx: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var tt = if (layout == CblasRowMajor) rowMajorTransChar(if (trans == CblasConjTrans) CblasTrans else trans) else transChar(trans);
    var dd = diagChar(diag);
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    f.internal_admitted.dtpmv_(&operation, &uu, &tt, &dd, &nn, ap, x, &ix);
}
pub export fn cblas_ctpmv(layout: c_int, uplo: c_int, trans: c_int, diag: c_int, n: c_int, ap: [*]const ComplexF32, x: [*]ComplexF32, incx: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
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
    f.internal_admitted.ctpmv_(&operation, &uu, &tt, &dd, &nn, ap, x, &ix);
}
pub export fn cblas_ztpmv(layout: c_int, uplo: c_int, trans: c_int, diag: c_int, n: c_int, ap: [*]const ComplexF64, x: [*]ComplexF64, incx: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
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
    f.internal_admitted.ztpmv_(&operation, &uu, &tt, &dd, &nn, ap, x, &ix);
}

pub export fn cblas_strsv(layout: c_int, uplo: c_int, trans: c_int, diag: c_int, n: c_int, a: [*]const f32, lda: c_int, x: [*]f32, incx: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var tt = if (layout == CblasRowMajor) rowMajorTransChar(if (trans == CblasConjTrans) CblasTrans else trans) else transChar(trans);
    var dd = diagChar(diag);
    var nn = asBlasInt(n);
    var aa = asBlasInt(lda);
    var ix = asBlasInt(incx);
    f.internal_admitted.strsv_(&operation, &uu, &tt, &dd, &nn, a, &aa, x, &ix);
}
pub export fn cblas_dtrsv(layout: c_int, uplo: c_int, trans: c_int, diag: c_int, n: c_int, a: [*]const f64, lda: c_int, x: [*]f64, incx: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var tt = if (layout == CblasRowMajor) rowMajorTransChar(if (trans == CblasConjTrans) CblasTrans else trans) else transChar(trans);
    var dd = diagChar(diag);
    var nn = asBlasInt(n);
    var aa = asBlasInt(lda);
    var ix = asBlasInt(incx);
    f.internal_admitted.dtrsv_(&operation, &uu, &tt, &dd, &nn, a, &aa, x, &ix);
}
pub export fn cblas_ctrsv(layout: c_int, uplo: c_int, trans: c_int, diag: c_int, n: c_int, a: [*]const ComplexF32, lda: c_int, x: [*]ComplexF32, incx: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
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
    f.internal_admitted.ctrsv_(&operation, &uu, &tt, &dd, &nn, a, &aa, x, &ix);
}
pub export fn cblas_ztrsv(layout: c_int, uplo: c_int, trans: c_int, diag: c_int, n: c_int, a: [*]const ComplexF64, lda: c_int, x: [*]ComplexF64, incx: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
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
    f.internal_admitted.ztrsv_(&operation, &uu, &tt, &dd, &nn, a, &aa, x, &ix);
}

pub export fn cblas_stbsv(layout: c_int, uplo: c_int, trans: c_int, diag: c_int, n: c_int, k: c_int, a: [*]const f32, lda: c_int, x: [*]f32, incx: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var tt = if (layout == CblasRowMajor) rowMajorTransChar(if (trans == CblasConjTrans) CblasTrans else trans) else transChar(trans);
    var dd = diagChar(diag);
    var nn = asBlasInt(n);
    var kk = asBlasInt(k);
    var aa = asBlasInt(lda);
    var ix = asBlasInt(incx);
    f.internal_admitted.stbsv_(&operation, &uu, &tt, &dd, &nn, &kk, a, &aa, x, &ix);
}
pub export fn cblas_dtbsv(layout: c_int, uplo: c_int, trans: c_int, diag: c_int, n: c_int, k: c_int, a: [*]const f64, lda: c_int, x: [*]f64, incx: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var tt = if (layout == CblasRowMajor) rowMajorTransChar(if (trans == CblasConjTrans) CblasTrans else trans) else transChar(trans);
    var dd = diagChar(diag);
    var nn = asBlasInt(n);
    var kk = asBlasInt(k);
    var aa = asBlasInt(lda);
    var ix = asBlasInt(incx);
    f.internal_admitted.dtbsv_(&operation, &uu, &tt, &dd, &nn, &kk, a, &aa, x, &ix);
}
pub export fn cblas_ctbsv(layout: c_int, uplo: c_int, trans: c_int, diag: c_int, n: c_int, k: c_int, a: [*]const ComplexF32, lda: c_int, x: [*]ComplexF32, incx: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
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
    f.internal_admitted.ctbsv_(&operation, &uu, &tt, &dd, &nn, &kk, a, &aa, x, &ix);
}
pub export fn cblas_ztbsv(layout: c_int, uplo: c_int, trans: c_int, diag: c_int, n: c_int, k: c_int, a: [*]const ComplexF64, lda: c_int, x: [*]ComplexF64, incx: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
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
    f.internal_admitted.ztbsv_(&operation, &uu, &tt, &dd, &nn, &kk, a, &aa, x, &ix);
}

pub export fn cblas_stpsv(layout: c_int, uplo: c_int, trans: c_int, diag: c_int, n: c_int, ap: [*]const f32, x: [*]f32, incx: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var tt = if (layout == CblasRowMajor) rowMajorTransChar(if (trans == CblasConjTrans) CblasTrans else trans) else transChar(trans);
    var dd = diagChar(diag);
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    f.internal_admitted.stpsv_(&operation, &uu, &tt, &dd, &nn, ap, x, &ix);
}
pub export fn cblas_dtpsv(layout: c_int, uplo: c_int, trans: c_int, diag: c_int, n: c_int, ap: [*]const f64, x: [*]f64, incx: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var tt = if (layout == CblasRowMajor) rowMajorTransChar(if (trans == CblasConjTrans) CblasTrans else trans) else transChar(trans);
    var dd = diagChar(diag);
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    f.internal_admitted.dtpsv_(&operation, &uu, &tt, &dd, &nn, ap, x, &ix);
}
pub export fn cblas_ctpsv(layout: c_int, uplo: c_int, trans: c_int, diag: c_int, n: c_int, ap: [*]const ComplexF32, x: [*]ComplexF32, incx: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
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
    f.internal_admitted.ctpsv_(&operation, &uu, &tt, &dd, &nn, ap, x, &ix);
}
pub export fn cblas_ztpsv(layout: c_int, uplo: c_int, trans: c_int, diag: c_int, n: c_int, ap: [*]const ComplexF64, x: [*]ComplexF64, incx: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
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
    f.internal_admitted.ztpsv_(&operation, &uu, &tt, &dd, &nn, ap, x, &ix);
}

pub export fn cblas_sger(layout: c_int, m: c_int, n: c_int, alpha: f32, x: [*]const f32, incx: c_int, y: [*]const f32, incy: c_int, a: [*]f32, lda: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    if (!validLayout(layout)) return;
    var mm = asBlasInt(m);
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    var aa = asBlasInt(lda);
    if (layout == CblasRowMajor) rowMajorGer(f32, mm, nn, alpha, x, ix, y, iy, a, aa, false) else f.internal_admitted.sger_(&operation, &mm, &nn, &alpha, x, &ix, y, &iy, a, &aa);
}

pub export fn cblas_dger(layout: c_int, m: c_int, n: c_int, alpha: f64, x: [*]const f64, incx: c_int, y: [*]const f64, incy: c_int, a: [*]f64, lda: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    if (!validLayout(layout)) return;
    var mm = asBlasInt(m);
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    var aa = asBlasInt(lda);
    if (layout == CblasRowMajor) rowMajorGer(f64, mm, nn, alpha, x, ix, y, iy, a, aa, false) else f.internal_admitted.dger_(&operation, &mm, &nn, &alpha, x, &ix, y, &iy, a, &aa);
}

pub export fn cblas_cgeru(layout: c_int, m: c_int, n: c_int, alpha: *const ComplexF32, x: [*]const ComplexF32, incx: c_int, y: [*]const ComplexF32, incy: c_int, a: [*]ComplexF32, lda: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    if (!validLayout(layout)) return;
    var mm = asBlasInt(m);
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    var aa = asBlasInt(lda);
    if (layout == CblasRowMajor) rowMajorGer(ComplexF32, mm, nn, alpha.*, x, ix, y, iy, a, aa, false) else f.internal_admitted.cgeru_(&operation, &mm, &nn, alpha, x, &ix, y, &iy, a, &aa);
}

pub export fn cblas_zgeru(layout: c_int, m: c_int, n: c_int, alpha: *const ComplexF64, x: [*]const ComplexF64, incx: c_int, y: [*]const ComplexF64, incy: c_int, a: [*]ComplexF64, lda: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    if (!validLayout(layout)) return;
    var mm = asBlasInt(m);
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    var aa = asBlasInt(lda);
    if (layout == CblasRowMajor) rowMajorGer(ComplexF64, mm, nn, alpha.*, x, ix, y, iy, a, aa, false) else f.internal_admitted.zgeru_(&operation, &mm, &nn, alpha, x, &ix, y, &iy, a, &aa);
}

pub export fn cblas_cgerc(layout: c_int, m: c_int, n: c_int, alpha: *const ComplexF32, x: [*]const ComplexF32, incx: c_int, y: [*]const ComplexF32, incy: c_int, a: [*]ComplexF32, lda: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    if (!validLayout(layout)) return;
    var mm = asBlasInt(m);
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    var aa = asBlasInt(lda);
    if (layout == CblasRowMajor) rowMajorGer(ComplexF32, mm, nn, alpha.*, x, ix, y, iy, a, aa, true) else f.internal_admitted.cgerc_(&operation, &mm, &nn, alpha, x, &ix, y, &iy, a, &aa);
}

pub export fn cblas_zgerc(layout: c_int, m: c_int, n: c_int, alpha: *const ComplexF64, x: [*]const ComplexF64, incx: c_int, y: [*]const ComplexF64, incy: c_int, a: [*]ComplexF64, lda: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    if (!validLayout(layout)) return;
    var mm = asBlasInt(m);
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    var aa = asBlasInt(lda);
    if (layout == CblasRowMajor) rowMajorGer(ComplexF64, mm, nn, alpha.*, x, ix, y, iy, a, aa, true) else f.internal_admitted.zgerc_(&operation, &mm, &nn, alpha, x, &ix, y, &iy, a, &aa);
}

pub export fn cblas_ssyr(layout: c_int, uplo: c_int, n: c_int, alpha: f32, x: [*]const f32, incx: c_int, a: [*]f32, lda: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var aa = asBlasInt(lda);
    f.internal_admitted.ssyr_(&operation, &uu, &nn, &alpha, x, &ix, a, &aa);
}
pub export fn cblas_dsyr(layout: c_int, uplo: c_int, n: c_int, alpha: f64, x: [*]const f64, incx: c_int, a: [*]f64, lda: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var aa = asBlasInt(lda);
    f.internal_admitted.dsyr_(&operation, &uu, &nn, &alpha, x, &ix, a, &aa);
}
pub export fn cblas_cher(layout: c_int, uplo: c_int, n: c_int, alpha: f32, x: [*]const ComplexF32, incx: c_int, a: [*]ComplexF32, lda: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var aa = asBlasInt(lda);
    f.internal_admitted.cher_(&operation, &uu, &nn, &alpha, x, &ix, a, &aa);
}
pub export fn cblas_zher(layout: c_int, uplo: c_int, n: c_int, alpha: f64, x: [*]const ComplexF64, incx: c_int, a: [*]ComplexF64, lda: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var aa = asBlasInt(lda);
    f.internal_admitted.zher_(&operation, &uu, &nn, &alpha, x, &ix, a, &aa);
}

pub export fn cblas_sspr(layout: c_int, uplo: c_int, n: c_int, alpha: f32, x: [*]const f32, incx: c_int, ap: [*]f32) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    f.internal_admitted.sspr_(&operation, &uu, &nn, &alpha, x, &ix, ap);
}
pub export fn cblas_dspr(layout: c_int, uplo: c_int, n: c_int, alpha: f64, x: [*]const f64, incx: c_int, ap: [*]f64) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    f.internal_admitted.dspr_(&operation, &uu, &nn, &alpha, x, &ix, ap);
}
pub export fn cblas_chpr(layout: c_int, uplo: c_int, n: c_int, alpha: f32, x: [*]const ComplexF32, incx: c_int, ap: [*]ComplexF32) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    f.internal_admitted.chpr_(&operation, &uu, &nn, &alpha, x, &ix, ap);
}
pub export fn cblas_zhpr(layout: c_int, uplo: c_int, n: c_int, alpha: f64, x: [*]const ComplexF64, incx: c_int, ap: [*]ComplexF64) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    f.internal_admitted.zhpr_(&operation, &uu, &nn, &alpha, x, &ix, ap);
}

pub export fn cblas_ssyr2(layout: c_int, uplo: c_int, n: c_int, alpha: f32, x: [*]const f32, incx: c_int, y: [*]const f32, incy: c_int, a: [*]f32, lda: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    var aa = asBlasInt(lda);
    f.internal_admitted.ssyr2_(&operation, &uu, &nn, &alpha, x, &ix, y, &iy, a, &aa);
}
pub export fn cblas_dsyr2(layout: c_int, uplo: c_int, n: c_int, alpha: f64, x: [*]const f64, incx: c_int, y: [*]const f64, incy: c_int, a: [*]f64, lda: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    var aa = asBlasInt(lda);
    f.internal_admitted.dsyr2_(&operation, &uu, &nn, &alpha, x, &ix, y, &iy, a, &aa);
}
pub export fn cblas_cher2(layout: c_int, uplo: c_int, n: c_int, alpha: *const ComplexF32, x: [*]const ComplexF32, incx: c_int, y: [*]const ComplexF32, incy: c_int, a: [*]ComplexF32, lda: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    var aa = asBlasInt(lda);
    f.internal_admitted.cher2_(&operation, &uu, &nn, alpha, x, &ix, y, &iy, a, &aa);
}
pub export fn cblas_zher2(layout: c_int, uplo: c_int, n: c_int, alpha: *const ComplexF64, x: [*]const ComplexF64, incx: c_int, y: [*]const ComplexF64, incy: c_int, a: [*]ComplexF64, lda: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    var aa = asBlasInt(lda);
    f.internal_admitted.zher2_(&operation, &uu, &nn, alpha, x, &ix, y, &iy, a, &aa);
}

pub export fn cblas_sspr2(layout: c_int, uplo: c_int, n: c_int, alpha: f32, x: [*]const f32, incx: c_int, y: [*]const f32, incy: c_int, ap: [*]f32) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.internal_admitted.sspr2_(&operation, &uu, &nn, &alpha, x, &ix, y, &iy, ap);
}
pub export fn cblas_dspr2(layout: c_int, uplo: c_int, n: c_int, alpha: f64, x: [*]const f64, incx: c_int, y: [*]const f64, incy: c_int, ap: [*]f64) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.internal_admitted.dspr2_(&operation, &uu, &nn, &alpha, x, &ix, y, &iy, ap);
}
pub export fn cblas_chpr2(layout: c_int, uplo: c_int, n: c_int, alpha: *const ComplexF32, x: [*]const ComplexF32, incx: c_int, y: [*]const ComplexF32, incy: c_int, ap: [*]ComplexF32) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.internal_admitted.chpr2_(&operation, &uu, &nn, alpha, x, &ix, y, &iy, ap);
}
pub export fn cblas_zhpr2(layout: c_int, uplo: c_int, n: c_int, alpha: *const ComplexF64, x: [*]const ComplexF64, incx: c_int, y: [*]const ComplexF64, incy: c_int, ap: [*]ComplexF64) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var nn = asBlasInt(n);
    var ix = asBlasInt(incx);
    var iy = asBlasInt(incy);
    f.internal_admitted.zhpr2_(&operation, &uu, &nn, alpha, x, &ix, y, &iy, ap);
}

// Level 3 CBLAS wrappers.
pub export fn cblas_sgemm(layout: c_int, transa: c_int, transb: c_int, m: c_int, n: c_int, k: c_int, alpha: f32, a: [*]const f32, lda: c_int, b: [*]const f32, ldb: c_int, beta: f32, c: [*]f32, ldc: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    if (!validLayout(layout) or !validTrans(transa) or !validTrans(transb)) return;
    var ta = if (layout == CblasRowMajor) transChar(transb) else transChar(transa);
    var tb = if (layout == CblasRowMajor) transChar(transa) else transChar(transb);
    var mm = asBlasInt(if (layout == CblasRowMajor) n else m);
    var nn = asBlasInt(if (layout == CblasRowMajor) m else n);
    var kk = asBlasInt(k);
    var la = asBlasInt(if (layout == CblasRowMajor) ldb else lda);
    var lb = asBlasInt(if (layout == CblasRowMajor) lda else ldb);
    var lc = asBlasInt(ldc);
    if (layout == CblasRowMajor) f.internal_admitted.sgemm_(&operation, &ta, &tb, &mm, &nn, &kk, &alpha, b, &la, a, &lb, &beta, c, &lc) else f.internal_admitted.sgemm_(&operation, &ta, &tb, &mm, &nn, &kk, &alpha, a, &la, b, &lb, &beta, c, &lc);
}

pub export fn cblas_dgemm(layout: c_int, transa: c_int, transb: c_int, m: c_int, n: c_int, k: c_int, alpha: f64, a: [*]const f64, lda: c_int, b: [*]const f64, ldb: c_int, beta: f64, c: [*]f64, ldc: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    if (!validLayout(layout) or !validTrans(transa) or !validTrans(transb)) return;
    var ta = if (layout == CblasRowMajor) transChar(transb) else transChar(transa);
    var tb = if (layout == CblasRowMajor) transChar(transa) else transChar(transb);
    var mm = asBlasInt(if (layout == CblasRowMajor) n else m);
    var nn = asBlasInt(if (layout == CblasRowMajor) m else n);
    var kk = asBlasInt(k);
    var la = asBlasInt(if (layout == CblasRowMajor) ldb else lda);
    var lb = asBlasInt(if (layout == CblasRowMajor) lda else ldb);
    var lc = asBlasInt(ldc);
    if (layout == CblasRowMajor) f.internal_admitted.dgemm_(&operation, &ta, &tb, &mm, &nn, &kk, &alpha, b, &la, a, &lb, &beta, c, &lc) else f.internal_admitted.dgemm_(&operation, &ta, &tb, &mm, &nn, &kk, &alpha, a, &la, b, &lb, &beta, c, &lc);
}

pub export fn cblas_cgemm(layout: c_int, transa: c_int, transb: c_int, m: c_int, n: c_int, k: c_int, alpha: *const ComplexF32, a: [*]const ComplexF32, lda: c_int, b: [*]const ComplexF32, ldb: c_int, beta: *const ComplexF32, c: [*]ComplexF32, ldc: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    if (!validLayout(layout) or !validTrans(transa) or !validTrans(transb)) return;
    var ta = if (layout == CblasRowMajor) transChar(transb) else transChar(transa);
    var tb = if (layout == CblasRowMajor) transChar(transa) else transChar(transb);
    var mm = asBlasInt(if (layout == CblasRowMajor) n else m);
    var nn = asBlasInt(if (layout == CblasRowMajor) m else n);
    var kk = asBlasInt(k);
    var la = asBlasInt(if (layout == CblasRowMajor) ldb else lda);
    var lb = asBlasInt(if (layout == CblasRowMajor) lda else ldb);
    var lc = asBlasInt(ldc);
    if (layout == CblasRowMajor) f.internal_admitted.cgemm_(&operation, &ta, &tb, &mm, &nn, &kk, alpha, b, &la, a, &lb, beta, c, &lc) else f.internal_admitted.cgemm_(&operation, &ta, &tb, &mm, &nn, &kk, alpha, a, &la, b, &lb, beta, c, &lc);
}

pub export fn cblas_zgemm(layout: c_int, transa: c_int, transb: c_int, m: c_int, n: c_int, k: c_int, alpha: *const ComplexF64, a: [*]const ComplexF64, lda: c_int, b: [*]const ComplexF64, ldb: c_int, beta: *const ComplexF64, c: [*]ComplexF64, ldc: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    if (!validLayout(layout) or !validTrans(transa) or !validTrans(transb)) return;
    var ta = if (layout == CblasRowMajor) transChar(transb) else transChar(transa);
    var tb = if (layout == CblasRowMajor) transChar(transa) else transChar(transb);
    var mm = asBlasInt(if (layout == CblasRowMajor) n else m);
    var nn = asBlasInt(if (layout == CblasRowMajor) m else n);
    var kk = asBlasInt(k);
    var la = asBlasInt(if (layout == CblasRowMajor) ldb else lda);
    var lb = asBlasInt(if (layout == CblasRowMajor) lda else ldb);
    var lc = asBlasInt(ldc);
    if (layout == CblasRowMajor) f.internal_admitted.zgemm_(&operation, &ta, &tb, &mm, &nn, &kk, alpha, b, &la, a, &lb, beta, c, &lc) else f.internal_admitted.zgemm_(&operation, &ta, &tb, &mm, &nn, &kk, alpha, a, &la, b, &lb, beta, c, &lc);
}

pub export fn cblas_ssymm(layout: c_int, side: c_int, uplo: c_int, m: c_int, n: c_int, alpha: f32, a: [*]const f32, lda: c_int, b: [*]const f32, ldb: c_int, beta: f32, c: [*]f32, ldc: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    if (!validLayout(layout)) return;
    var ss = if (layout == CblasRowMajor) rowMajorSideChar(side) else sideChar(side);
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var mm = asBlasInt(if (layout == CblasRowMajor) n else m);
    var nn = asBlasInt(if (layout == CblasRowMajor) m else n);
    var la = asBlasInt(lda);
    var lb = asBlasInt(ldb);
    var lc = asBlasInt(ldc);
    f.internal_admitted.ssymm_(&operation, &ss, &uu, &mm, &nn, &alpha, a, &la, b, &lb, &beta, c, &lc);
}

pub export fn cblas_dsymm(layout: c_int, side: c_int, uplo: c_int, m: c_int, n: c_int, alpha: f64, a: [*]const f64, lda: c_int, b: [*]const f64, ldb: c_int, beta: f64, c: [*]f64, ldc: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    if (!validLayout(layout)) return;
    var ss = if (layout == CblasRowMajor) rowMajorSideChar(side) else sideChar(side);
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var mm = asBlasInt(if (layout == CblasRowMajor) n else m);
    var nn = asBlasInt(if (layout == CblasRowMajor) m else n);
    var la = asBlasInt(lda);
    var lb = asBlasInt(ldb);
    var lc = asBlasInt(ldc);
    f.internal_admitted.dsymm_(&operation, &ss, &uu, &mm, &nn, &alpha, a, &la, b, &lb, &beta, c, &lc);
}

pub export fn cblas_csymm(layout: c_int, side: c_int, uplo: c_int, m: c_int, n: c_int, alpha: *const ComplexF32, a: [*]const ComplexF32, lda: c_int, b: [*]const ComplexF32, ldb: c_int, beta: *const ComplexF32, c: [*]ComplexF32, ldc: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    if (!validLayout(layout)) return;
    var ss = if (layout == CblasRowMajor) rowMajorSideChar(side) else sideChar(side);
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var mm = asBlasInt(if (layout == CblasRowMajor) n else m);
    var nn = asBlasInt(if (layout == CblasRowMajor) m else n);
    var la = asBlasInt(lda);
    var lb = asBlasInt(ldb);
    var lc = asBlasInt(ldc);
    f.internal_admitted.csymm_(&operation, &ss, &uu, &mm, &nn, alpha, a, &la, b, &lb, beta, c, &lc);
}

pub export fn cblas_zsymm(layout: c_int, side: c_int, uplo: c_int, m: c_int, n: c_int, alpha: *const ComplexF64, a: [*]const ComplexF64, lda: c_int, b: [*]const ComplexF64, ldb: c_int, beta: *const ComplexF64, c: [*]ComplexF64, ldc: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    if (!validLayout(layout)) return;
    var ss = if (layout == CblasRowMajor) rowMajorSideChar(side) else sideChar(side);
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var mm = asBlasInt(if (layout == CblasRowMajor) n else m);
    var nn = asBlasInt(if (layout == CblasRowMajor) m else n);
    var la = asBlasInt(lda);
    var lb = asBlasInt(ldb);
    var lc = asBlasInt(ldc);
    f.internal_admitted.zsymm_(&operation, &ss, &uu, &mm, &nn, alpha, a, &la, b, &lb, beta, c, &lc);
}

pub export fn cblas_chemm(layout: c_int, side: c_int, uplo: c_int, m: c_int, n: c_int, alpha: *const ComplexF32, a: [*]const ComplexF32, lda: c_int, b: [*]const ComplexF32, ldb: c_int, beta: *const ComplexF32, c: [*]ComplexF32, ldc: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    if (!validLayout(layout)) return;
    var ss = if (layout == CblasRowMajor) rowMajorSideChar(side) else sideChar(side);
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var mm = asBlasInt(if (layout == CblasRowMajor) n else m);
    var nn = asBlasInt(if (layout == CblasRowMajor) m else n);
    var la = asBlasInt(lda);
    var lb = asBlasInt(ldb);
    var lc = asBlasInt(ldc);
    f.internal_admitted.chemm_(&operation, &ss, &uu, &mm, &nn, alpha, a, &la, b, &lb, beta, c, &lc);
}

pub export fn cblas_zhemm(layout: c_int, side: c_int, uplo: c_int, m: c_int, n: c_int, alpha: *const ComplexF64, a: [*]const ComplexF64, lda: c_int, b: [*]const ComplexF64, ldb: c_int, beta: *const ComplexF64, c: [*]ComplexF64, ldc: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    if (!validLayout(layout)) return;
    var ss = if (layout == CblasRowMajor) rowMajorSideChar(side) else sideChar(side);
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var mm = asBlasInt(if (layout == CblasRowMajor) n else m);
    var nn = asBlasInt(if (layout == CblasRowMajor) m else n);
    var la = asBlasInt(lda);
    var lb = asBlasInt(ldb);
    var lc = asBlasInt(ldc);
    f.internal_admitted.zhemm_(&operation, &ss, &uu, &mm, &nn, alpha, a, &la, b, &lb, beta, c, &lc);
}

pub export fn cblas_ssyrk(layout: c_int, uplo: c_int, trans: c_int, n: c_int, k: c_int, alpha: f32, a: [*]const f32, lda: c_int, beta: f32, c: [*]f32, ldc: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var tt = if (layout == CblasRowMajor) rowMajorTransChar(if (trans == CblasConjTrans) CblasTrans else trans) else transChar(trans);
    var nn = asBlasInt(n);
    var kk = asBlasInt(k);
    var la = asBlasInt(lda);
    var lc = asBlasInt(ldc);
    f.internal_admitted.ssyrk_(&operation, &uu, &tt, &nn, &kk, &alpha, a, &la, &beta, c, &lc);
}

pub export fn cblas_dsyrk(layout: c_int, uplo: c_int, trans: c_int, n: c_int, k: c_int, alpha: f64, a: [*]const f64, lda: c_int, beta: f64, c: [*]f64, ldc: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var tt = if (layout == CblasRowMajor) rowMajorTransChar(if (trans == CblasConjTrans) CblasTrans else trans) else transChar(trans);
    var nn = asBlasInt(n);
    var kk = asBlasInt(k);
    var la = asBlasInt(lda);
    var lc = asBlasInt(ldc);
    f.internal_admitted.dsyrk_(&operation, &uu, &tt, &nn, &kk, &alpha, a, &la, &beta, c, &lc);
}

pub export fn cblas_csyrk(layout: c_int, uplo: c_int, trans: c_int, n: c_int, k: c_int, alpha: *const ComplexF32, a: [*]const ComplexF32, lda: c_int, beta: *const ComplexF32, c: [*]ComplexF32, ldc: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var tt = if (layout == CblasRowMajor) rowMajorTransChar(trans) else transChar(trans);
    var nn = asBlasInt(n);
    var kk = asBlasInt(k);
    var la = asBlasInt(lda);
    var lc = asBlasInt(ldc);
    f.internal_admitted.csyrk_(&operation, &uu, &tt, &nn, &kk, alpha, a, &la, beta, c, &lc);
}

pub export fn cblas_zsyrk(layout: c_int, uplo: c_int, trans: c_int, n: c_int, k: c_int, alpha: *const ComplexF64, a: [*]const ComplexF64, lda: c_int, beta: *const ComplexF64, c: [*]ComplexF64, ldc: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var tt = if (layout == CblasRowMajor) rowMajorTransChar(trans) else transChar(trans);
    var nn = asBlasInt(n);
    var kk = asBlasInt(k);
    var la = asBlasInt(lda);
    var lc = asBlasInt(ldc);
    f.internal_admitted.zsyrk_(&operation, &uu, &tt, &nn, &kk, alpha, a, &la, beta, c, &lc);
}

pub export fn cblas_cherk(layout: c_int, uplo: c_int, trans: c_int, n: c_int, k: c_int, alpha: f32, a: [*]const ComplexF32, lda: c_int, beta: f32, c: [*]ComplexF32, ldc: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var tt = if (layout == CblasRowMajor) rowMajorTransChar(trans) else transChar(trans);
    var nn = asBlasInt(n);
    var kk = asBlasInt(k);
    var la = asBlasInt(lda);
    var lc = asBlasInt(ldc);
    f.internal_admitted.cherk_(&operation, &uu, &tt, &nn, &kk, &alpha, a, &la, &beta, c, &lc);
}

pub export fn cblas_zherk(layout: c_int, uplo: c_int, trans: c_int, n: c_int, k: c_int, alpha: f64, a: [*]const ComplexF64, lda: c_int, beta: f64, c: [*]ComplexF64, ldc: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var tt = if (layout == CblasRowMajor) rowMajorTransChar(trans) else transChar(trans);
    var nn = asBlasInt(n);
    var kk = asBlasInt(k);
    var la = asBlasInt(lda);
    var lc = asBlasInt(ldc);
    f.internal_admitted.zherk_(&operation, &uu, &tt, &nn, &kk, &alpha, a, &la, &beta, c, &lc);
}

pub export fn cblas_ssyr2k(layout: c_int, uplo: c_int, trans: c_int, n: c_int, k: c_int, alpha: f32, a: [*]const f32, lda: c_int, b: [*]const f32, ldb: c_int, beta: f32, c: [*]f32, ldc: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var tt = if (layout == CblasRowMajor) rowMajorTransChar(if (trans == CblasConjTrans) CblasTrans else trans) else transChar(trans);
    var nn = asBlasInt(n);
    var kk = asBlasInt(k);
    var la = asBlasInt(lda);
    var lb = asBlasInt(ldb);
    var lc = asBlasInt(ldc);
    f.internal_admitted.ssyr2k_(&operation, &uu, &tt, &nn, &kk, &alpha, a, &la, b, &lb, &beta, c, &lc);
}

pub export fn cblas_dsyr2k(layout: c_int, uplo: c_int, trans: c_int, n: c_int, k: c_int, alpha: f64, a: [*]const f64, lda: c_int, b: [*]const f64, ldb: c_int, beta: f64, c: [*]f64, ldc: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var tt = if (layout == CblasRowMajor) rowMajorTransChar(if (trans == CblasConjTrans) CblasTrans else trans) else transChar(trans);
    var nn = asBlasInt(n);
    var kk = asBlasInt(k);
    var la = asBlasInt(lda);
    var lb = asBlasInt(ldb);
    var lc = asBlasInt(ldc);
    f.internal_admitted.dsyr2k_(&operation, &uu, &tt, &nn, &kk, &alpha, a, &la, b, &lb, &beta, c, &lc);
}

pub export fn cblas_csyr2k(layout: c_int, uplo: c_int, trans: c_int, n: c_int, k: c_int, alpha: *const ComplexF32, a: [*]const ComplexF32, lda: c_int, b: [*]const ComplexF32, ldb: c_int, beta: *const ComplexF32, c: [*]ComplexF32, ldc: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var tt = if (layout == CblasRowMajor) rowMajorTransChar(trans) else transChar(trans);
    var nn = asBlasInt(n);
    var kk = asBlasInt(k);
    var la = asBlasInt(lda);
    var lb = asBlasInt(ldb);
    var lc = asBlasInt(ldc);
    f.internal_admitted.csyr2k_(&operation, &uu, &tt, &nn, &kk, alpha, a, &la, b, &lb, beta, c, &lc);
}

pub export fn cblas_zsyr2k(layout: c_int, uplo: c_int, trans: c_int, n: c_int, k: c_int, alpha: *const ComplexF64, a: [*]const ComplexF64, lda: c_int, b: [*]const ComplexF64, ldb: c_int, beta: *const ComplexF64, c: [*]ComplexF64, ldc: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var tt = if (layout == CblasRowMajor) rowMajorTransChar(trans) else transChar(trans);
    var nn = asBlasInt(n);
    var kk = asBlasInt(k);
    var la = asBlasInt(lda);
    var lb = asBlasInt(ldb);
    var lc = asBlasInt(ldc);
    f.internal_admitted.zsyr2k_(&operation, &uu, &tt, &nn, &kk, alpha, a, &la, b, &lb, beta, c, &lc);
}

pub export fn cblas_cher2k(layout: c_int, uplo: c_int, trans: c_int, n: c_int, k: c_int, alpha: *const ComplexF32, a: [*]const ComplexF32, lda: c_int, b: [*]const ComplexF32, ldb: c_int, beta: f32, c: [*]ComplexF32, ldc: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var tt = if (layout == CblasRowMajor) rowMajorTransChar(trans) else transChar(trans);
    var nn = asBlasInt(n);
    var kk = asBlasInt(k);
    var la = asBlasInt(lda);
    var lb = asBlasInt(ldb);
    var lc = asBlasInt(ldc);
    f.internal_admitted.cher2k_(&operation, &uu, &tt, &nn, &kk, alpha, a, &la, b, &lb, &beta, c, &lc);
}

pub export fn cblas_zher2k(layout: c_int, uplo: c_int, trans: c_int, n: c_int, k: c_int, alpha: *const ComplexF64, a: [*]const ComplexF64, lda: c_int, b: [*]const ComplexF64, ldb: c_int, beta: f64, c: [*]ComplexF64, ldc: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    if (!validLayout(layout)) return;
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var tt = if (layout == CblasRowMajor) rowMajorTransChar(trans) else transChar(trans);
    var nn = asBlasInt(n);
    var kk = asBlasInt(k);
    var la = asBlasInt(lda);
    var lb = asBlasInt(ldb);
    var lc = asBlasInt(ldc);
    f.internal_admitted.zher2k_(&operation, &uu, &tt, &nn, &kk, alpha, a, &la, b, &lb, &beta, c, &lc);
}

pub export fn cblas_strmm(layout: c_int, side: c_int, uplo: c_int, transa: c_int, diag: c_int, m: c_int, n: c_int, alpha: f32, a: [*]const f32, lda: c_int, b: [*]f32, ldb: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    if (!validLayout(layout) or !validSide(side) or !validUplo(uplo) or !validTrans(transa) or !validDiag(diag)) return;
    var ss = if (layout == CblasRowMajor) rowMajorSideChar(side) else sideChar(side);
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var tt = transChar(transa);
    var dd = diagChar(diag);
    var mm = asBlasInt(if (layout == CblasRowMajor) n else m);
    var nn = asBlasInt(if (layout == CblasRowMajor) m else n);
    var la = asBlasInt(lda);
    var lb = asBlasInt(ldb);
    f.internal_admitted.strmm_(&operation, &ss, &uu, &tt, &dd, &mm, &nn, &alpha, a, &la, b, &lb);
}

pub export fn cblas_dtrmm(layout: c_int, side: c_int, uplo: c_int, transa: c_int, diag: c_int, m: c_int, n: c_int, alpha: f64, a: [*]const f64, lda: c_int, b: [*]f64, ldb: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    if (!validLayout(layout) or !validSide(side) or !validUplo(uplo) or !validTrans(transa) or !validDiag(diag)) return;
    var ss = if (layout == CblasRowMajor) rowMajorSideChar(side) else sideChar(side);
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var tt = transChar(transa);
    var dd = diagChar(diag);
    var mm = asBlasInt(if (layout == CblasRowMajor) n else m);
    var nn = asBlasInt(if (layout == CblasRowMajor) m else n);
    var la = asBlasInt(lda);
    var lb = asBlasInt(ldb);
    f.internal_admitted.dtrmm_(&operation, &ss, &uu, &tt, &dd, &mm, &nn, &alpha, a, &la, b, &lb);
}

pub export fn cblas_ctrmm(layout: c_int, side: c_int, uplo: c_int, transa: c_int, diag: c_int, m: c_int, n: c_int, alpha: *const ComplexF32, a: [*]const ComplexF32, lda: c_int, b: [*]ComplexF32, ldb: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    if (!validLayout(layout) or !validSide(side) or !validUplo(uplo) or !validTrans(transa) or !validDiag(diag)) return;
    var ss = if (layout == CblasRowMajor) rowMajorSideChar(side) else sideChar(side);
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var tt = transChar(transa);
    var dd = diagChar(diag);
    var mm = asBlasInt(if (layout == CblasRowMajor) n else m);
    var nn = asBlasInt(if (layout == CblasRowMajor) m else n);
    var la = asBlasInt(lda);
    var lb = asBlasInt(ldb);
    f.internal_admitted.ctrmm_(&operation, &ss, &uu, &tt, &dd, &mm, &nn, alpha, a, &la, b, &lb);
}

pub export fn cblas_ztrmm(layout: c_int, side: c_int, uplo: c_int, transa: c_int, diag: c_int, m: c_int, n: c_int, alpha: *const ComplexF64, a: [*]const ComplexF64, lda: c_int, b: [*]ComplexF64, ldb: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    if (!validLayout(layout) or !validSide(side) or !validUplo(uplo) or !validTrans(transa) or !validDiag(diag)) return;
    var ss = if (layout == CblasRowMajor) rowMajorSideChar(side) else sideChar(side);
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var tt = transChar(transa);
    var dd = diagChar(diag);
    var mm = asBlasInt(if (layout == CblasRowMajor) n else m);
    var nn = asBlasInt(if (layout == CblasRowMajor) m else n);
    var la = asBlasInt(lda);
    var lb = asBlasInt(ldb);
    f.internal_admitted.ztrmm_(&operation, &ss, &uu, &tt, &dd, &mm, &nn, alpha, a, &la, b, &lb);
}

pub export fn cblas_strsm(layout: c_int, side: c_int, uplo: c_int, transa: c_int, diag: c_int, m: c_int, n: c_int, alpha: f32, a: [*]const f32, lda: c_int, b: [*]f32, ldb: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    if (!validLayout(layout) or !validSide(side) or !validUplo(uplo) or !validTrans(transa) or !validDiag(diag)) return;
    var ss = if (layout == CblasRowMajor) rowMajorSideChar(side) else sideChar(side);
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var tt = transChar(transa);
    var dd = diagChar(diag);
    var mm = asBlasInt(if (layout == CblasRowMajor) n else m);
    var nn = asBlasInt(if (layout == CblasRowMajor) m else n);
    var la = asBlasInt(lda);
    var lb = asBlasInt(ldb);
    f.internal_admitted.strsm_(&operation, &ss, &uu, &tt, &dd, &mm, &nn, &alpha, a, &la, b, &lb);
}

pub export fn cblas_dtrsm(layout: c_int, side: c_int, uplo: c_int, transa: c_int, diag: c_int, m: c_int, n: c_int, alpha: f64, a: [*]const f64, lda: c_int, b: [*]f64, ldb: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    if (!validLayout(layout) or !validSide(side) or !validUplo(uplo) or !validTrans(transa) or !validDiag(diag)) return;
    var ss = if (layout == CblasRowMajor) rowMajorSideChar(side) else sideChar(side);
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var tt = transChar(transa);
    var dd = diagChar(diag);
    var mm = asBlasInt(if (layout == CblasRowMajor) n else m);
    var nn = asBlasInt(if (layout == CblasRowMajor) m else n);
    var la = asBlasInt(lda);
    var lb = asBlasInt(ldb);
    f.internal_admitted.dtrsm_(&operation, &ss, &uu, &tt, &dd, &mm, &nn, &alpha, a, &la, b, &lb);
}

pub export fn cblas_ctrsm(layout: c_int, side: c_int, uplo: c_int, transa: c_int, diag: c_int, m: c_int, n: c_int, alpha: *const ComplexF32, a: [*]const ComplexF32, lda: c_int, b: [*]ComplexF32, ldb: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    if (!validLayout(layout) or !validSide(side) or !validUplo(uplo) or !validTrans(transa) or !validDiag(diag)) return;
    var ss = if (layout == CblasRowMajor) rowMajorSideChar(side) else sideChar(side);
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var tt = transChar(transa);
    var dd = diagChar(diag);
    var mm = asBlasInt(if (layout == CblasRowMajor) n else m);
    var nn = asBlasInt(if (layout == CblasRowMajor) m else n);
    var la = asBlasInt(lda);
    var lb = asBlasInt(ldb);
    f.internal_admitted.ctrsm_(&operation, &ss, &uu, &tt, &dd, &mm, &nn, alpha, a, &la, b, &lb);
}

pub export fn cblas_ztrsm(layout: c_int, side: c_int, uplo: c_int, transa: c_int, diag: c_int, m: c_int, n: c_int, alpha: *const ComplexF64, a: [*]const ComplexF64, lda: c_int, b: [*]ComplexF64, ldb: c_int) callconv(.c) void {
    var operation = execution.beginDefault();
    defer operation.end();
    if (!validLayout(layout) or !validSide(side) or !validUplo(uplo) or !validTrans(transa) or !validDiag(diag)) return;
    var ss = if (layout == CblasRowMajor) rowMajorSideChar(side) else sideChar(side);
    var uu = if (layout == CblasRowMajor) rowMajorUploChar(uplo) else uploChar(uplo);
    var tt = transChar(transa);
    var dd = diagChar(diag);
    var mm = asBlasInt(if (layout == CblasRowMajor) n else m);
    var nn = asBlasInt(if (layout == CblasRowMajor) m else n);
    var la = asBlasInt(lda);
    var lb = asBlasInt(ldb);
    f.internal_admitted.ztrsm_(&operation, &ss, &uu, &tt, &dd, &mm, &nn, alpha, a, &la, b, &lb);
}
