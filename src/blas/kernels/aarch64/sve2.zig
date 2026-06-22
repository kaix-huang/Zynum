// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const gemm_task = @import("../gemm_task.zig");
const asimd = @import("asimd.zig");
const features = @import("features.zig");

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
