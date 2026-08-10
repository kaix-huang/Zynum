// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Typed main-graph facade for the isolated x86_64 Level 2 width object.

const abi = @import("x86_64_level2_width_abi.zig");
const types = @import("../../types.zig");

const BlasInt = types.BlasInt;

extern fn zynum_internal_x86_64_level2_width_execute(request: *abi.Request) callconv(.c) u8;

fn scalarTag(comptime T: type) abi.Scalar {
    if (T == f32) return .f32;
    if (T == f64) return .f64;
    if (T == types.ComplexF32) return .complex_f32;
    if (T == types.ComplexF64) return .complex_f64;
    @compileError("isolated Level 2 width bridge supports BLAS real and complex scalars");
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

fn execute(
    comptime T: type,
    operation: abi.Operation,
    flags: u8,
    m: usize,
    n: usize,
    alpha: T,
    beta: T,
    input0: *const anyopaque,
    input1: *const anyopaque,
    output: *anyopaque,
    lda: BlasInt,
) bool {
    const alpha_parts = components(T, alpha);
    const beta_parts = components(T, beta);
    var request: abi.Request = .{
        .operation = @intFromEnum(operation),
        .scalar = @intFromEnum(scalarTag(T)),
        .flags = flags,
        .lda = lda,
        .m = m,
        .n = n,
        .alpha_re = alpha_parts[0],
        .alpha_im = alpha_parts[1],
        .beta_re = beta_parts[0],
        .beta_im = beta_parts[1],
        .input0 = input0,
        .input1 = input1,
        .output = output,
    };
    return zynum_internal_x86_64_level2_width_execute(&request) != 0;
}

pub noinline fn tryGemvNoTransUnit(
    comptime T: type,
    m: usize,
    n: usize,
    alpha: T,
    a: [*]const T,
    lda: BlasInt,
    x: [*]const T,
    y: [*]T,
) bool {
    const operation: abi.Operation = if (T == f32 or T == f64) .gemv_no_trans_unit else .gemv_no_trans_complex;
    return execute(T, operation, 0, m, n, alpha, alpha, @ptrCast(a), @ptrCast(x), @ptrCast(y), lda);
}

pub noinline fn tryGemvNoTransFull(
    comptime T: type,
    m: usize,
    n: usize,
    alpha: T,
    a: [*]const T,
    lda: BlasInt,
    x: [*]const T,
    beta: T,
    y: [*]T,
) bool {
    return execute(T, .gemv_no_trans_full, 0, m, n, alpha, beta, @ptrCast(a), @ptrCast(x), @ptrCast(y), lda);
}

pub noinline fn tryGemvTransUnit(
    comptime T: type,
    m: usize,
    n: usize,
    alpha: T,
    a: [*]const T,
    lda: BlasInt,
    x: [*]const T,
    y: [*]T,
    conjugate: bool,
) bool {
    const operation: abi.Operation = if (T == f32 or T == f64) .gemv_trans_unit else .gemv_trans_complex;
    return execute(T, operation, @intFromBool(conjugate), m, n, alpha, alpha, @ptrCast(a), @ptrCast(x), @ptrCast(y), lda);
}

pub noinline fn tryGemvTransFull(
    comptime T: type,
    m: usize,
    n: usize,
    alpha: T,
    a: [*]const T,
    lda: BlasInt,
    x: [*]const T,
    beta: T,
    y: [*]T,
) bool {
    return execute(T, .gemv_trans_full, 0, m, n, alpha, beta, @ptrCast(a), @ptrCast(x), @ptrCast(y), lda);
}

pub noinline fn tryGer(
    comptime T: type,
    m: usize,
    n: usize,
    alpha: T,
    x: [*]const T,
    y: [*]const T,
    a: [*]T,
    lda: BlasInt,
    conjugate_y: bool,
) bool {
    const operation: abi.Operation = if (T == f32 or T == f64) .ger_real else .ger_complex;
    return execute(T, operation, @intFromBool(conjugate_y), m, n, alpha, alpha, @ptrCast(x), @ptrCast(y), @ptrCast(a), lda);
}
