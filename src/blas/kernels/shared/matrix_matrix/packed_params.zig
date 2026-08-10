// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Shared packed-SIMD GEMM tile parameters.
//!
//! Keep architecture wrappers and catalog descriptors in sync without importing
//! executable architecture modules into the catalog.

pub const k_unroll = 4;
pub const max_stack_pack_bytes = 64 * 1024;

pub const PackedSimdShape = struct {
    lane_count: comptime_int,
    tile_n: comptime_int,
    row_groups: comptime_int,
    tail_vector_lanes: comptime_int,
    k_unroll: comptime_int,
    max_stack_pack_bytes: comptime_int,
};

pub fn aarch64AsimdShape(comptime T: type) PackedSimdShape {
    if (T == f32) {
        return .{
            .lane_count = 4,
            .tile_n = 8,
            .row_groups = 3,
            .tail_vector_lanes = 2,
            .k_unroll = k_unroll,
            .max_stack_pack_bytes = max_stack_pack_bytes,
        };
    }
    if (T == f64) {
        return .{
            .lane_count = 2,
            .tile_n = 8,
            .row_groups = 3,
            .tail_vector_lanes = 0,
            .k_unroll = k_unroll,
            .max_stack_pack_bytes = max_stack_pack_bytes,
        };
    }
    @compileError("AArch64 ASIMD packed GEMM supports f32 and f64");
}

pub fn x86Shape(comptime T: type, comptime has_avx: bool, comptime has_avx512f: bool) PackedSimdShape {
    if (T == f32) {
        return .{
            .lane_count = if (has_avx512f) 16 else if (has_avx) 8 else 4,
            .tile_n = if (has_avx) 12 else 8,
            .row_groups = if (has_avx512f) 2 else 1,
            .tail_vector_lanes = if (has_avx512f) 8 else if (has_avx) 4 else 0,
            .k_unroll = k_unroll,
            .max_stack_pack_bytes = max_stack_pack_bytes,
        };
    }
    if (T == f64) {
        return .{
            .lane_count = if (has_avx512f) 8 else if (has_avx) 4 else 2,
            .tile_n = if (has_avx) 8 else 6,
            .row_groups = if (has_avx512f) 2 else 1,
            .tail_vector_lanes = if (has_avx512f) 4 else if (has_avx) 2 else 0,
            .k_unroll = k_unroll,
            .max_stack_pack_bytes = max_stack_pack_bytes,
        };
    }
    @compileError("x86_64 packed GEMM supports f32 and f64");
}
