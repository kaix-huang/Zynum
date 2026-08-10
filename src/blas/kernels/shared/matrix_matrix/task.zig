// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const types = @import("../../../types.zig");
const catalog = @import("catalog.zig");

pub const BlasInt = types.BlasInt;

pub const AppleAmxKernelId = enum {
    none,
    apple_amx_f32_n16,
    apple_amx_f32_n32,
    apple_amx_f64_n8,
    apple_amx_f64_n16,
    apple_amx_f64_n32,
};

pub const BPackPath = enum {
    natural,
    dynamic,
    transpose4,
};

pub const BLayout = enum {
    no_trans,
    trans,
};

pub const SmeF32Panel = enum {
    panels2x2,
    panels2x2_u4,
};

pub const PackWorkspacePlan = struct {
    stack_bytes: usize = 0,
    cache_bytes: usize = 0,
};

pub const ExecutionPlan = struct {
    selected_kernel: catalog.KernelId = .auto,
    fallback_kernel: catalog.KernelId = .auto,
    amx: AppleAmxKernelId = .none,
    amx_partial_n16: bool = false,
    b_pack: BPackPath = .natural,
    f32_panel: SmeF32Panel = .panels2x2,
    sme_panel_batch: usize = 1,
    pack: PackWorkspacePlan = .{},
    amx_pack: PackWorkspacePlan = .{},
};

pub fn Task(comptime T: type) type {
    return struct {
        m: usize,
        n0: usize,
        n1: usize,
        k: usize,
        alpha: T,
        a: [*]const T,
        lda: BlasInt,
        b: [*]const T,
        ldb: BlasInt,
        b_layout: BLayout = .no_trans,
        beta: T,
        c: [*]T,
        ldc: BlasInt,
        allow_sme: bool = false,
        kernel: catalog.KernelId = .auto,
        execution: ExecutionPlan = .{},
    };
}

pub inline fn toUsize(x: BlasInt) usize {
    return @intCast(x);
}

pub inline fn matIndex(lda: BlasInt, row: usize, col: usize) usize {
    return row + col * toUsize(lda);
}
