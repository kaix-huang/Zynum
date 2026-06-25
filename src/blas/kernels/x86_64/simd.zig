// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const features = @import("features.zig");
const packed_simd = @import("../matrix_matrix/packed_simd.zig");
const gemm_task = @import("../matrix_matrix/task.zig");

pub const enabled: bool = features.has_sse2;
pub const supports_avx: bool = features.has_avx;
pub const supports_avx2: bool = features.has_avx2;
pub const supports_avx512f: bool = features.has_avx512f;
pub const supports_fma: bool = features.has_fma;

const max_stack_pack_bytes = 64 * 1024;

fn lanes(comptime T: type) comptime_int {
    if (T == f32) {
        if (comptime features.has_avx512f) return 16;
        if (comptime features.has_avx) return 8;
        return 4;
    }
    if (T == f64) {
        if (comptime features.has_avx512f) return 8;
        if (comptime features.has_avx) return 4;
        return 2;
    }
    @compileError("x86_64 SIMD GEMM kernels support f32 and f64");
}

fn columnTile(comptime T: type) comptime_int {
    if (T == f32) {
        if (comptime features.has_avx) return 12;
        return 8;
    }
    if (T == f64) {
        if (comptime features.has_avx) return 8;
        return 6;
    }
    @compileError("x86_64 SIMD GEMM kernels support f32 and f64");
}

fn config(comptime T: type) packed_simd.Config {
    return .{
        .lane_count = lanes(T),
        .tile_n = columnTile(T),
        .row_groups = 1,
        .k_unroll = 4,
        .max_stack_pack_bytes = max_stack_pack_bytes,
        .pack_tail_columns = false,
    };
}

pub fn preferredColumnBlock(comptime T: type) usize {
    return columnTile(T);
}

pub fn noTransRealF32(task: gemm_task.Task(f32)) void {
    packed_simd.noTransReal(f32, config(f32), task);
}

pub fn noTransRealF64(task: gemm_task.Task(f64)) void {
    packed_simd.noTransReal(f64, config(f64), task);
}
