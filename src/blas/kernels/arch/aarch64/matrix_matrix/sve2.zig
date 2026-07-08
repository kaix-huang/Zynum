// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const gemm_task = @import("../../../shared/matrix_matrix/task.zig");
const asimd = @import("asimd.zig");
const features = @import("../features.zig");

// SVE2 feature-gated GEMM descriptor that currently executes the ASIMD/FMA
// packed kernel. Keep backend names explicit (`sve2_asimd`) until true scalable
// SVE2 GEMM kernels replace this wrapper.

pub const enabled: bool = features.has_sve2;

pub fn vectorBytes() usize {
    return features.sveVectorBytes();
}

pub fn preferredColumnBlock(comptime T: type) usize {
    return asimd.preferredColumnBlock(T);
}

pub fn noTransRealF32(task: gemm_task.Task(f32)) void {
    asimd.noTransRealF32(task);
}

pub fn noTransRealF64(task: gemm_task.Task(f64)) void {
    asimd.noTransRealF64(task);
}
