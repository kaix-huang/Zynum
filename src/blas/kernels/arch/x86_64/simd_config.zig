// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Shared x86_64 fixed-width SIMD kernel parameters.

const features = @import("features.zig");
const matrix_vector_simd = @import("../../shared/matrix_vector/fixed_simd.zig");
const types = @import("../../../types.zig");
const vector_simd = @import("../../shared/vector/fixed_simd.zig");

pub const enabled: bool = features.has_sse2;

pub fn realType(comptime T: type) type {
    if (T == f32 or T == types.ComplexF32) return f32;
    if (T == f64 or T == types.ComplexF64) return f64;
    @compileError("x86_64 fixed SIMD kernels support f32, f64, ComplexF32, and ComplexF64");
}

pub fn lanes(comptime T: type) comptime_int {
    const R = realType(T);
    if (R == f32) {
        if (comptime features.has_avx512f) return 16;
        if (comptime features.has_avx) return 8;
        return 4;
    }
    if (comptime features.has_avx512f) return 8;
    if (comptime features.has_avx) return 4;
    return 2;
}

pub fn vectorUnrollVectors() comptime_int {
    return if (features.has_avx512f) 6 else 4;
}

pub fn copyLaneCount() comptime_int {
    return if (features.has_avx512f) 128 else if (features.has_avx) 64 else 32;
}

pub fn vectorConfig(comptime T: type) vector_simd.Config {
    return .{
        .lane_count = lanes(T),
        .unroll_vectors = vectorUnrollVectors(),
        .copy_lane_count = copyLaneCount(),
    };
}

pub fn asumVectorConfig(comptime T: type) vector_simd.Config {
    const R = realType(T);
    return .{
        .lane_count = lanes(T),
        .unroll_vectors = if (R == f64 and features.has_avx512f) 8 else vectorUnrollVectors(),
        .copy_lane_count = copyLaneCount(),
    };
}

pub const byte_config = vector_simd.Config{
    .lane_count = 4,
    .unroll_vectors = vectorUnrollVectors(),
    .copy_lane_count = copyLaneCount(),
};

pub fn matrixColumnUnroll(comptime T: type) comptime_int {
    _ = realType(T);
    return if (features.has_avx) 8 else 4;
}

pub fn matrixRowUnrollVectors() comptime_int {
    return if (features.has_avx512f) 3 else 4;
}

pub fn matrixConfig(comptime T: type) matrix_vector_simd.Config {
    return .{
        .lane_count = lanes(T),
        .row_unroll_vectors = matrixRowUnrollVectors(),
        .col_unroll = matrixColumnUnroll(T),
        .min_work = 0,
        .max_work = 512 * 512,
    };
}

pub fn matrixPackedRowsConfig(comptime T: type) matrix_vector_simd.Config {
    return .{
        .lane_count = lanes(T),
        .row_unroll_vectors = matrixRowUnrollVectors(),
        .col_unroll = matrixColumnUnroll(T),
    };
}

pub fn matrixComplexConfig(comptime T: type) matrix_vector_simd.Config {
    const R = realType(T);
    if (T != types.ComplexF32 and T != types.ComplexF64) {
        @compileError("x86_64 complex matrix-vector kernels support ComplexF32 and ComplexF64");
    }
    return .{
        .lane_count = lanes(R),
        .row_unroll_vectors = matrixRowUnrollVectors(),
        .col_unroll = if (R == f32) 4 else 2,
        .min_work = 128 * 128,
        .max_work = 512 * 512,
    };
}
