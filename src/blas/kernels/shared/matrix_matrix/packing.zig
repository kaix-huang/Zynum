// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Shared GEMM packing helpers for K-major, panel-contiguous B panels.

const gemm_task = @import("task.zig");

const matIndex = gemm_task.matIndex;

pub inline fn packBPanel(comptime T: type, comptime panel_cols: usize, task: gemm_task.Task(T), j: usize, b_pack: []T) void {
    var p: usize = 0;
    while (p < task.k) : (p += 1) {
        const base = p * panel_cols;
        inline for (0..panel_cols) |col| {
            b_pack[base + col] = task.b[matIndex(task.ldb, p, j + col)];
        }
    }
}

pub fn packBPanelDynamic(comptime T: type, task: gemm_task.Task(T), j: usize, panel_cols: usize, b_pack: []T) void {
    var p: usize = 0;
    while (p < task.k) : (p += 1) {
        const base = p * panel_cols;
        var col: usize = 0;
        while (col < panel_cols) : (col += 1) {
            b_pack[base + col] = task.b[matIndex(task.ldb, p, j + col)];
        }
    }
}

pub fn packBPanelWithDefault(comptime T: type, comptime default_panel_cols: usize, task: gemm_task.Task(T), j: usize, panel_cols: usize, b_pack: []T) void {
    switch (panel_cols) {
        default_panel_cols => packBPanel(T, default_panel_cols, task, j, b_pack),
        else => packBPanelDynamic(T, task, j, panel_cols, b_pack),
    }
}
