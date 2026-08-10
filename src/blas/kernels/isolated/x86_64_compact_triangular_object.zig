// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Separately compiled x86_64 compact-triangular implementation object.

const abi = @import("x86_64_compact_triangular_abi.zig");
const packed_unit = @import("../../core/matrix_vector/triangular_packed_unit.zig");
const band_solve = @import("../../core/matrix_vector/triangular_band_solve.zig");
const scalar = @import("../../core/shared/scalar.zig");

extern var zynum_internal_x86_64_compact_triangular_enabled: u8;

fn coreUplo(value: abi.Uplo) scalar.Uplo {
    return switch (value) {
        .upper => .upper,
        .lower => .lower,
    };
}

fn coreTranspose(value: abi.Transpose) scalar.Order {
    return switch (value) {
        .no_trans => .no_trans,
        .trans => .trans,
        .conj_trans => .conj_trans,
    };
}

fn coreDiagonal(value: abi.Diagonal) scalar.Diag {
    return switch (value) {
        .unit => .unit,
        .non_unit => .non_unit,
    };
}

fn matrixPtr(comptime T: type, request: *const abi.Request) [*]const T {
    return @ptrCast(@alignCast(request.matrix));
}

fn vectorPtr(comptime T: type, request: *const abi.Request) [*]T {
    return @ptrCast(@alignCast(request.vector));
}

fn dispatchScalar(comptime T: type, operation: abi.Operation, request: *abi.Request) bool {
    const uplo = coreUplo(@enumFromInt(request.uplo));
    const trans = coreTranspose(@enumFromInt(request.transpose));
    const diag = coreDiagonal(@enumFromInt(request.diagonal));
    return switch (operation) {
        .tpmv => packed_unit.tryTpmv(T, uplo, trans, diag, @intCast(request.n), matrixPtr(T, request), vectorPtr(T, request), request.incx),
        .tpsv => packed_unit.tryTpsv(T, uplo, trans, diag, @intCast(request.n), matrixPtr(T, request), vectorPtr(T, request), request.incx),
        .tbsv => band_solve.tryTbsv(T, uplo, trans, diag, request.n, request.k, matrixPtr(T, request), request.lda, vectorPtr(T, request), request.incx),
    };
}

fn execute(request: *abi.Request) callconv(.c) u8 {
    if (zynum_internal_x86_64_compact_triangular_enabled == 0) return 0;
    const operation: abi.Operation = @enumFromInt(request.operation);
    const scalar_tag: abi.Scalar = @enumFromInt(request.scalar);
    const handled = switch (scalar_tag) {
        .f32 => dispatchScalar(f32, operation, request),
        .f64 => dispatchScalar(f64, operation, request),
        .complex_f32 => dispatchScalar(scalar.ComplexF32, operation, request),
        .complex_f64 => dispatchScalar(scalar.ComplexF64, operation, request),
    };
    return @intFromBool(handled);
}

comptime {
    @export(&execute, .{
        .name = "zynum_internal_x86_64_compact_triangular_execute",
        .visibility = .hidden,
    });
}
