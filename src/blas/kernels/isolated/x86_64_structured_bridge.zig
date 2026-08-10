// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Main-graph typed facade for isolated structured Level 3 candidates.

const abi = @import("x86_64_structured_abi.zig");
const gemm_host = @import("structured_gemm_host.zig");
const task_runtime_host = @import("task_runtime_host.zig");
const scalar = @import("../../core/shared/scalar.zig");

const BlasInt = scalar.BlasInt;
const Side = scalar.Side;
const Uplo = scalar.Uplo;
const Order = scalar.Order;
const Diag = scalar.Diag;

extern fn zynum_internal_x86_64_structured_execute(request: *abi.Request) callconv(.c) u8;

fn scalarTag(comptime T: type) abi.Scalar {
    if (T == f32) return .f32;
    if (T == f64) return .f64;
    if (T == scalar.ComplexF32) return .complex_f32;
    if (T == scalar.ComplexF64) return .complex_f64;
    @compileError("isolated structured bridge supports BLAS scalar types");
}

fn bits(comptime T: type, value: T) u64 {
    if (T == f32) return @as(u32, @bitCast(value));
    if (T == f64) return @bitCast(value);
    @compileError("bits expects a real scalar component");
}

fn components(comptime T: type, value: T) [2]u64 {
    if (T == f32 or T == f64) return .{ bits(T, value), 0 };
    return .{ bits(@TypeOf(value.re), value.re), bits(@TypeOf(value.im), value.im) };
}

fn sideTag(value: Side) abi.Side {
    return if (value == .left) .left else .right;
}

fn uploTag(value: Uplo) abi.Uplo {
    return if (value == .upper) .upper else .lower;
}

fn transposeTag(value: Order) abi.Transpose {
    return switch (value) {
        .no_trans => .no_trans,
        .trans => .trans,
        .conj_trans => .conj_trans,
    };
}

fn diagonalTag(value: Diag) abi.Diagonal {
    return if (value == .unit) .unit else .non_unit;
}

fn execute(comptime T: type, operation: abi.Operation, side: Side, uplo: Uplo, transpose: Order, diagonal: Diag, flags: u8, m: BlasInt, n: BlasInt, alpha: T, beta: T, a: [*]const T, lda: BlasInt, b: [*]const T, ldb: BlasInt, c: [*]T, ldc: BlasInt) bool {
    _ = gemm_host;
    _ = task_runtime_host;
    const alpha_parts = components(T, alpha);
    const beta_parts = components(T, beta);
    var request: abi.Request = .{
        .operation = @intFromEnum(operation),
        .scalar = @intFromEnum(scalarTag(T)),
        .side = @intFromEnum(sideTag(side)),
        .uplo = @intFromEnum(uploTag(uplo)),
        .transpose = @intFromEnum(transposeTag(transpose)),
        .diagonal = @intFromEnum(diagonalTag(diagonal)),
        .flags = flags,
        .m = m,
        .n = n,
        .lda = lda,
        .ldb = ldb,
        .ldc = ldc,
        .alpha_re = alpha_parts[0],
        .alpha_im = alpha_parts[1],
        .beta_re = beta_parts[0],
        .beta_im = beta_parts[1],
        .a = @ptrCast(a),
        .b = @ptrCast(b),
        .c = @ptrCast(c),
    };
    return zynum_internal_x86_64_structured_execute(&request) != 0;
}

pub noinline fn trySymm(comptime T: type, side: Side, uplo: Uplo, m: BlasInt, n: BlasInt, alpha: T, a: [*]const T, lda: BlasInt, b: [*]const T, ldb: BlasInt, beta: T, c: [*]T, ldc: BlasInt, hermitian: bool) bool {
    return execute(T, .symm_dense, side, uplo, .no_trans, .non_unit, @intFromBool(hermitian), m, n, alpha, beta, a, lda, b, ldb, c, ldc);
}

pub noinline fn tryTrmmRight(comptime T: type, uplo: Uplo, transpose: Order, diagonal: Diag, m: BlasInt, n: BlasInt, alpha: T, a: [*]const T, lda: BlasInt, b: [*]T, ldb: BlasInt) bool {
    return execute(T, .trmm_right, .right, uplo, transpose, diagonal, 0, m, n, alpha, scalar.zero(T), a, lda, b, ldb, b, ldb);
}

pub noinline fn tryTrsmRight(comptime T: type, uplo: Uplo, transpose: Order, diagonal: Diag, m: BlasInt, n: BlasInt, alpha: T, a: [*]const T, lda: BlasInt, b: [*]T, ldb: BlasInt) bool {
    return execute(T, .trsm_right, .right, uplo, transpose, diagonal, 0, m, n, alpha, scalar.zero(T), a, lda, b, ldb, b, ldb);
}
