// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Separately compiled AVX-512-hosted, AVX2-width Level 2 bodies.

const abi = @import("x86_64_level2_width_abi.zig");
const simd_config = @import("../arch/x86_64/simd_config.zig");
const fixed_simd = @import("../shared/matrix_vector/fixed_simd.zig");
const types = @import("../../types.zig");

extern var zynum_internal_x86_64_level2_width_enabled: u8;

fn inputPtr(comptime T: type, pointer: *const anyopaque) [*]const T {
    return @ptrCast(@alignCast(pointer));
}

fn outputPtr(comptime T: type, pointer: *anyopaque) [*]T {
    return @ptrCast(@alignCast(pointer));
}

fn realValue(comptime T: type, raw: u64) T {
    if (T == f32) return @bitCast(@as(u32, @truncate(raw)));
    if (T == f64) return @bitCast(raw);
    @compileError("realValue expects f32 or f64");
}

fn scalarValue(comptime T: type, re: u64, im: u64) T {
    if (T == f32 or T == f64) return realValue(T, re);
    const Component = @TypeOf(@as(T, undefined).re);
    return .{ .re = realValue(Component, re), .im = realValue(Component, im) };
}

fn dispatchReal(comptime T: type, operation: abi.Operation, request: *abi.Request) bool {
    if (T != f32) return false;
    const cfg = comptime simd_config.matrixNarrowConfig(T);
    const alpha = scalarValue(T, request.alpha_re, request.alpha_im);
    const beta = scalarValue(T, request.beta_re, request.beta_im);
    const a = inputPtr(T, request.input0);
    const x = inputPtr(T, request.input1);
    const y = outputPtr(T, request.output);
    return switch (operation) {
        .gemv_no_trans_unit => fixed_simd.gemvNoTransUnitReal(T, cfg, request.m, request.n, alpha, a, request.lda, x, y),
        .gemv_no_trans_full => fixed_simd.gemvNoTransFullUnitReal(T, cfg, request.m, request.n, alpha, a, request.lda, x, beta, y),
        .gemv_trans_unit => fixed_simd.gemvTransUnitReal(T, cfg, request.m, request.n, alpha, a, request.lda, x, y),
        .gemv_trans_full => fixed_simd.gemvTransFullUnitReal(T, cfg, request.m, request.n, alpha, a, request.lda, x, beta, y),
        .ger_real => fixed_simd.gerUnitReal(T, cfg, request.m, request.n, alpha, a, x, y, request.lda),
        else => false,
    };
}

fn dispatchComplex(comptime T: type, operation: abi.Operation, request: *abi.Request) bool {
    const cfg = comptime simd_config.matrixComplexNarrowConfig(T);
    const alpha = scalarValue(T, request.alpha_re, request.alpha_im);
    const input0 = inputPtr(T, request.input0);
    const input1 = inputPtr(T, request.input1);
    const output = outputPtr(T, request.output);
    return switch (operation) {
        .gemv_no_trans_complex => fixed_simd.gemvNoTransUnitComplex(T, cfg, request.m, request.n, alpha, input0, request.lda, input1, output),
        .gemv_trans_complex => fixed_simd.gemvTransUnitComplex(T, cfg, request.m, request.n, alpha, input0, request.lda, input1, output, request.flags != 0),
        .ger_complex => fixed_simd.gerUnitComplex(T, cfg, request.m, request.n, alpha, input0, input1, output, request.lda, request.flags != 0),
        else => false,
    };
}

fn execute(request: *abi.Request) callconv(.c) u8 {
    if (zynum_internal_x86_64_level2_width_enabled == 0 or comptime !simd_config.has_avx512_width) return 0;
    const operation: abi.Operation = @enumFromInt(request.operation);
    const scalar: abi.Scalar = @enumFromInt(request.scalar);
    const handled = switch (scalar) {
        .f32 => dispatchReal(f32, operation, request),
        .f64 => false,
        .complex_f32 => dispatchComplex(types.ComplexF32, operation, request),
        .complex_f64 => dispatchComplex(types.ComplexF64, operation, request),
    };
    return @intFromBool(handled);
}

comptime {
    @export(&execute, .{
        .name = "zynum_internal_x86_64_level2_width_execute",
        .visibility = .hidden,
    });
}
