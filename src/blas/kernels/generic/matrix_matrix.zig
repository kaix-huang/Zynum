// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const basic = @import("matrix_matrix/basic.zig");
const gemm_task = @import("../matrix_matrix/task.zig");

pub fn preferredColumnBlock(comptime T: type) usize {
    if (T == f32 or T == f64) return 4;
    @compileError("generic GEMM kernels support f32 and f64");
}

pub fn noTransReal(comptime T: type, task: gemm_task.Task(T)) void {
    if (T == f32) {
        noTransRealF32(task);
    } else if (T == f64) {
        noTransRealF64(task);
    } else {
        @compileError("generic GEMM kernels support f32 and f64");
    }
}

pub fn noTransRealF32(task: gemm_task.Task(f32)) void {
    const n = task.n1 - task.n0;
    if (task.m == 1 and n >= 8 and task.k >= 16) return basic.f32x4x8(task);
    if (task.m == 1 and n >= 4 and task.k >= 16) return basic.f32x4x4(task);
    if (task.m == 1 and task.k >= 16) return basic.rowVectorColumns(f32, task);
    if (n == 1 and task.m >= 8 and task.k >= 16) return basic.f32x8x1(task);
    if (task.m >= 4 and n >= 8 and task.k >= 8) return basic.f32x4x8(task);
    if (task.m >= 4 and n >= 4) return basic.f32x4x4(task);
    basic.f32x4x1(task);
}

pub fn noTransRealF64(task: gemm_task.Task(f64)) void {
    const n = task.n1 - task.n0;
    if (task.m == 1 and n >= 6 and task.k >= 16) return basic.f64x2x6(task);
    if (task.m == 1 and n >= 4 and task.k >= 16) return basic.f64x2x4(task);
    if (task.m == 1 and task.k >= 16) return basic.rowVectorColumns(f64, task);
    if (n == 1 and task.m >= 4 and task.k >= 16) return basic.f64x4x1(task);
    if (task.m >= 2 and n >= 6 and task.k >= 8) return basic.f64x2x6(task);
    if (task.m >= 2 and n >= 4) return basic.f64x2x4(task);
    basic.f64x2x1(task);
}
