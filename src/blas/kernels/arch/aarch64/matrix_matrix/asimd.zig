// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const features = @import("../features.zig");
const packed_params = @import("../../../shared/matrix_matrix/packed_params.zig");
const packed_simd = @import("../../../shared/matrix_matrix/packed_simd.zig");
const gemm_task = @import("../../../shared/matrix_matrix/task.zig");

pub const enabled: bool = features.has_asimd;

fn shape(comptime T: type) packed_params.PackedSimdShape {
    return packed_params.aarch64AsimdShape(T);
}

fn config(comptime T: type) packed_simd.Config {
    const s = shape(T);
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

fn lowKF64Config() packed_simd.Config {
    const s = shape(f64);
    return .{
        .lane_count = s.lane_count,
        .tile_n = s.tile_n,
        .row_groups = 2,
        .k_unroll = s.k_unroll,
        .tail_vector_lanes = s.tail_vector_lanes,
        .max_stack_pack_bytes = s.max_stack_pack_bytes,
        .pack_tail_columns = true,
        .special_low_k_pack = true,
    };
}

fn lowKF64WideNConfig() packed_simd.Config {
    const s = shape(f64);
    return .{
        .lane_count = s.lane_count,
        .tile_n = 12,
        .row_groups = 2,
        .k_unroll = s.k_unroll,
        .tail_vector_lanes = s.tail_vector_lanes,
        .max_stack_pack_bytes = s.max_stack_pack_bytes,
        .pack_tail_columns = true,
    };
}

fn lowKF32WideNConfig(comptime k_unroll: comptime_int) packed_simd.Config {
    const s = shape(f32);
    return .{
        .lane_count = s.lane_count,
        .tile_n = 6,
        .row_groups = 4,
        .k_unroll = k_unroll,
        .tail_vector_lanes = s.tail_vector_lanes,
        .max_stack_pack_bytes = s.max_stack_pack_bytes,
        .pack_tail_columns = true,
        .special_low_k_pack = k_unroll == 8,
    };
}

fn useLowKF32WideNConfig(task: gemm_task.Task(f32)) bool {
    const n = task.n1 - task.n0;
    return task.k <= 33 and task.m >= 48 and n >= 48 and n > task.m;
}

fn useLowKF64Config(task: gemm_task.Task(f64)) bool {
    return task.k <= 33 and task.m >= 48 and task.n1 - task.n0 >= 48;
}

fn useLowKF64WideNConfig(task: gemm_task.Task(f64)) bool {
    const n = task.n1 - task.n0;
    return task.k <= 17 and useLowKF64Config(task) and n > task.m;
}

pub fn preferredColumnBlock(comptime T: type) usize {
    return shape(T).tile_n;
}

noinline fn noTransRealF32LowKWideK4(task: gemm_task.Task(f32)) void {
    packed_simd.noTransReal(f32, lowKF32WideNConfig(4), task);
}

noinline fn noTransRealF32LowKWideK8(task: gemm_task.Task(f32)) void {
    packed_simd.noTransReal(f32, lowKF32WideNConfig(8), task);
}

pub fn noTransRealF32(task: gemm_task.Task(f32)) void {
    if (useLowKF32WideNConfig(task)) {
        if (task.k >= 31) return noTransRealF32LowKWideK8(task);
        return noTransRealF32LowKWideK4(task);
    }
    packed_simd.noTransReal(f32, config(f32), task);
}

pub fn noTransRealF64(task: gemm_task.Task(f64)) void {
    if (useLowKF64WideNConfig(task)) return packed_simd.noTransReal(f64, lowKF64WideNConfig(), task);
    if (useLowKF64Config(task)) return packed_simd.noTransReal(f64, lowKF64Config(), task);
    packed_simd.noTransReal(f64, config(f64), task);
}
