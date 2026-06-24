// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const features = @import("features.zig");
const packed_simd = @import("../matrix_matrix/packed_simd.zig");
const gemm_task = @import("../matrix_matrix/task.zig");

const f32_tile_n = 8;
const f64_tile_n = 8;
const max_stack_pack_bytes = 64 * 1024;

pub const enabled: bool = features.has_asimd;

fn lanes(comptime T: type) comptime_int {
    if (T == f32) return 4;
    if (T == f64) return 2;
    @compileError("ASIMD GEMM kernels support f32 and f64");
}

fn columnTile(comptime T: type) comptime_int {
    if (T == f32) return f32_tile_n;
    if (T == f64) return f64_tile_n;
    @compileError("ASIMD GEMM kernels support f32 and f64");
}

fn rowGroups(comptime T: type) comptime_int {
    if (T == f32 or T == f64) return 3;
    @compileError("ASIMD GEMM kernels support f32 and f64");
}

fn tailVectorLanes(comptime T: type) comptime_int {
    if (T == f32) return 2;
    if (T == f64) return 0;
    @compileError("ASIMD GEMM kernels support f32 and f64");
}

fn config(comptime T: type) packed_simd.Config {
    return .{
        .lane_count = lanes(T),
        .tile_n = columnTile(T),
        .row_groups = rowGroups(T),
        .k_unroll = 4,
        .tail_vector_lanes = tailVectorLanes(T),
        .max_stack_pack_bytes = max_stack_pack_bytes,
        .pack_tail_columns = true,
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
