// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Shared AArch64 fixed-width ASIMD kernel parameters.

const matrix_vector_simd = @import("../../shared/matrix_vector/fixed_simd.zig");
const types = @import("../../../types.zig");
const vector_simd = @import("../../shared/vector/fixed_simd.zig");

pub fn realType(comptime T: type) type {
    if (T == f32 or T == types.ComplexF32) return f32;
    if (T == f64 or T == types.ComplexF64) return f64;
    @compileError("AArch64 fixed SIMD kernels support f32, f64, ComplexF32, and ComplexF64");
}

pub fn asimdLanes(comptime T: type) comptime_int {
    return if (realType(T) == f32) 4 else 2;
}

pub fn vectorConfig(comptime T: type) vector_simd.Config {
    return .{
        .lane_count = asimdLanes(T),
        .unroll_vectors = 4,
        .copy_lane_count = 64,
    };
}

pub const byte_config = vector_simd.Config{
    .lane_count = 4,
    .unroll_vectors = 4,
    .copy_lane_count = 64,
};

pub fn matrixConfig(comptime T: type) matrix_vector_simd.Config {
    if (T != f32 and T != f64) {
        @compileError("AArch64 fixed matrix-vector kernels support f32 and f64");
    }
    return .{
        .lane_count = asimdLanes(T),
        .row_unroll_vectors = 4,
        .col_unroll = if (T == f32) 8 else 4,
        .max_work = 512 * 512,
    };
}

pub fn matrixComplexConfig(comptime T: type) matrix_vector_simd.Config {
    if (T != types.ComplexF32 and T != types.ComplexF64) {
        @compileError("AArch64 fixed complex matrix-vector kernels support ComplexF32 and ComplexF64");
    }
    const R = realType(T);
    return .{
        .lane_count = asimdLanes(T),
        .row_unroll_vectors = 4,
        .col_unroll = if (R == f32) 4 else 2,
        .min_work = 128 * 128,
        .max_work = 512 * 512,
    };
}
