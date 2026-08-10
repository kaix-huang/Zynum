// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Main-graph typed facade for the isolated x86_64 compact-triangular object.

const abi = @import("x86_64_compact_triangular_abi.zig");
const scalar = @import("../../core/shared/scalar.zig");

const BlasInt = scalar.BlasInt;
const Order = scalar.Order;
const Uplo = scalar.Uplo;
const Diag = scalar.Diag;

extern fn zynum_internal_x86_64_compact_triangular_execute(request: *abi.Request) callconv(.c) u8;

fn scalarTag(comptime T: type) abi.Scalar {
    if (T == f32) return .f32;
    if (T == f64) return .f64;
    if (T == scalar.ComplexF32) return .complex_f32;
    if (T == scalar.ComplexF64) return .complex_f64;
    @compileError("isolated compact-triangular bridge supports BLAS real and complex scalars");
}

fn uploTag(value: Uplo) abi.Uplo {
    return switch (value) {
        .upper => .upper,
        .lower => .lower,
    };
}

fn transposeTag(value: Order) abi.Transpose {
    return switch (value) {
        .no_trans => .no_trans,
        .trans => .trans,
        .conj_trans => .conj_trans,
    };
}

fn diagonalTag(value: Diag) abi.Diagonal {
    return switch (value) {
        .unit => .unit,
        .non_unit => .non_unit,
    };
}

fn execute(
    comptime T: type,
    operation: abi.Operation,
    uplo: Uplo,
    trans: Order,
    diag: Diag,
    n: BlasInt,
    k: BlasInt,
    matrix: [*]const T,
    lda: BlasInt,
    vector: [*]T,
    incx: BlasInt,
) bool {
    var request = abi.init(
        operation,
        scalarTag(T),
        uploTag(uplo),
        transposeTag(trans),
        diagonalTag(diag),
        n,
        k,
        lda,
        @ptrCast(matrix),
        @ptrCast(vector),
        incx,
    );
    return zynum_internal_x86_64_compact_triangular_execute(&request) != 0;
}

pub noinline fn tryTpmv(
    comptime T: type,
    uplo: Uplo,
    trans: Order,
    diag: Diag,
    n: BlasInt,
    ap: [*]const T,
    vector: [*]T,
    incx: BlasInt,
) bool {
    return execute(T, .tpmv, uplo, trans, diag, n, 0, ap, 0, vector, incx);
}

pub noinline fn tryTpsv(
    comptime T: type,
    uplo: Uplo,
    trans: Order,
    diag: Diag,
    n: BlasInt,
    ap: [*]const T,
    vector: [*]T,
    incx: BlasInt,
) bool {
    return execute(T, .tpsv, uplo, trans, diag, n, 0, ap, 0, vector, incx);
}

pub noinline fn tryTbsv(
    comptime T: type,
    uplo: Uplo,
    trans: Order,
    diag: Diag,
    n: BlasInt,
    k: BlasInt,
    a: [*]const T,
    lda: BlasInt,
    vector: [*]T,
    incx: BlasInt,
) bool {
    return execute(T, .tbsv, uplo, trans, diag, n, k, a, lda, vector, incx);
}
