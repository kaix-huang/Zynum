// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Shared x86_64 fixed-width SIMD kernel parameters.

const features = @import("features.zig");
const contract = @import("../../contract.zig");
const matrix_vector_simd = @import("../../shared/matrix_vector/fixed_simd.zig");
const types = @import("../../../types.zig");
const vector_simd = @import("../../shared/vector/fixed_simd.zig");

pub const enabled: bool = features.has_sse2;
pub const has_avx512_width: bool = features.has_avx512f;

/// Stable identity of the fixed-SIMD specialization selected by this build.
/// Zynum currently uses target-tier builds rather than a runtime x86 fat
/// binary, so lower tiers are forced by compiling a separate artifact.
pub const capability: ?contract.IsaCapability = if (!enabled)
    null
else if (features.has_avx512f and features.has_fma)
    .x86_64_avx512f_fma
else if (features.has_avx512f)
    .x86_64_avx512f
else if (features.has_avx2 and features.has_fma)
    .x86_64_avx2_fma
else if (features.has_avx2)
    .x86_64_avx2
else if (features.has_fma)
    .x86_64_fma
else if (features.has_avx)
    .x86_64_avx
else
    .x86_64_sse2;

test "x86 fixed-SIMD build capability is present exactly when the tier is enabled" {
    const std = @import("std");
    try std.testing.expectEqual(enabled, capability != null);
}

test "AVX2-width Level 1 geometry remains independent of the build maximum" {
    const std = @import("std");
    const f32_cfg = avx2WidthVectorConfig(f32);
    const f64_cfg = avx2WidthVectorConfig(f64);
    try std.testing.expectEqual(@as(comptime_int, 8), f32_cfg.lane_count);
    try std.testing.expectEqual(@as(comptime_int, 4), f64_cfg.lane_count);
    try std.testing.expectEqual(@as(comptime_int, 4), f32_cfg.unroll_vectors);
    try std.testing.expectEqual(@as(comptime_int, 4), f64_cfg.unroll_vectors);
}

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

/// 256-bit/FMA-family geometry used as an independently benchmarked leaf in
/// AVX-512 builds. The active profile decides which operation families prefer
/// this geometry; lower-tier builds continue using `vectorConfig` directly.
pub fn avx2WidthVectorConfig(comptime T: type) vector_simd.Config {
    const R = realType(T);
    return .{
        .lane_count = if (R == f32) 8 else 4,
        .unroll_vectors = 4,
        .copy_lane_count = 64,
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

pub fn matrixNarrowConfig(comptime T: type) matrix_vector_simd.Config {
    const R = realType(T);
    return .{
        .lane_count = if (R == f32) 8 else 4,
        .row_unroll_vectors = 4,
        .col_unroll = 8,
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

pub fn matrixComplexNarrowConfig(comptime T: type) matrix_vector_simd.Config {
    const R = realType(T);
    if (T != types.ComplexF32 and T != types.ComplexF64) {
        @compileError("x86_64 complex matrix-vector kernels support ComplexF32 and ComplexF64");
    }
    return .{
        .lane_count = if (R == f32) 8 else 4,
        .row_unroll_vectors = 4,
        .col_unroll = if (R == f32) 4 else 2,
        .min_work = 128 * 128,
        .max_work = 512 * 512,
    };
}

pub fn matrixBodyConfig(comptime T: type) matrix_vector_simd.Config {
    return .{
        .lane_count = lanes(T),
        .row_unroll_vectors = matrixRowUnrollVectors(),
        .col_unroll = 1,
    };
}
