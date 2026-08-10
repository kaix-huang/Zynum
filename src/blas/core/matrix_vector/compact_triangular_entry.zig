// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Core dispatch entries for packed and banded triangular operations.

const builtin = @import("builtin");

const scalar = @import("../shared/scalar.zig");
const triangular = @import("triangular.zig");
const isolated = @import("../../kernels/isolated/x86_64_compact_triangular_bridge.zig");

const BlasInt = scalar.BlasInt;
const Order = scalar.Order;
const Uplo = scalar.Uplo;
const Diag = scalar.Diag;

pub noinline fn tpmv(comptime T: type, uplo: Uplo, trans_: Order, diag: Diag, n: BlasInt, ap: [*]const T, x: [*]T, incx: BlasInt) void {
    if (comptime builtin.cpu.arch == .x86_64) {
        if (n > 0 and incx != 0 and isolated.tryTpmv(T, uplo, trans_, diag, n, ap, x, incx)) return;
    }
    triangular.tpmv(T, uplo, trans_, diag, n, ap, x, incx);
}

pub noinline fn tpsv(comptime T: type, uplo: Uplo, trans_: Order, diag: Diag, n: BlasInt, ap: [*]const T, x: [*]T, incx: BlasInt) void {
    if (comptime builtin.cpu.arch == .x86_64) {
        if (n > 0 and incx != 0 and isolated.tryTpsv(T, uplo, trans_, diag, n, ap, x, incx)) return;
    }
    triangular.tpsv(T, uplo, trans_, diag, n, ap, x, incx);
}

pub noinline fn tbsv(comptime T: type, uplo: Uplo, trans_: Order, diag: Diag, n: BlasInt, k: BlasInt, a: [*]const T, lda: BlasInt, x: [*]T, incx: BlasInt) void {
    if (comptime builtin.cpu.arch == .x86_64) {
        if (n > 0 and incx != 0 and k >= 0 and isolated.tryTbsv(T, uplo, trans_, diag, n, k, a, lda, x, incx)) return;
    }
    triangular.tbsv(T, uplo, trans_, diag, n, k, a, lda, x, incx);
}
