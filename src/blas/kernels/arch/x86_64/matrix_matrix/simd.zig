// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const features = @import("../features.zig");
const packed_params = @import("../../../shared/matrix_matrix/packed_params.zig");
const packed_simd = @import("../../../shared/matrix_matrix/packed_simd.zig");
const gemm_task = @import("../../../shared/matrix_matrix/task.zig");

pub const enabled: bool = features.has_sse2;
pub const supports_avx: bool = features.has_avx;
pub const supports_avx2: bool = features.has_avx2;
pub const supports_avx512f: bool = features.has_avx512f;
pub const supports_fma: bool = features.has_fma;

pub const Tier = enum {
    sse2,
    avx,
    avx2_fma,
    avx512f_fma,
};

pub fn tierAvailable(comptime tier: Tier) bool {
    return switch (tier) {
        .sse2 => enabled,
        .avx => supports_avx,
        .avx2_fma => supports_avx2 and supports_fma,
        .avx512f_fma => supports_avx512f and supports_fma,
    };
}

fn shape(comptime T: type, comptime tier: Tier) packed_params.PackedSimdShape {
    return packed_params.x86Shape(T, tier != .sse2, tier == .avx512f_fma);
}

fn config(comptime T: type, comptime tier: Tier) packed_simd.Config {
    const s = shape(T, tier);
    return .{
        .lane_count = s.lane_count,
        .tile_n = s.tile_n,
        .row_groups = s.row_groups,
        .k_unroll = s.k_unroll,
        .tail_vector_lanes = s.tail_vector_lanes,
        .max_stack_pack_bytes = s.max_stack_pack_bytes,
        .pack_tail_columns = true,
    };
}

pub fn noTransRealF32For(comptime tier: Tier, task: gemm_task.Task(f32)) void {
    packed_simd.noTransReal(f32, config(f32, tier), task);
}

pub fn noTransRealF64For(comptime tier: Tier, task: gemm_task.Task(f64)) void {
    packed_simd.noTransReal(f64, config(f64, tier), task);
}
