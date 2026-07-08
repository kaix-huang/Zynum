// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const types = @import("../../../types.zig");
const catalog = @import("catalog.zig");

pub const BlasInt = types.BlasInt;

pub const Implementation = enum {
    auto,
    generic,
    arch_simd,
    streaming_matrix,
};

pub const AmxKernel = enum {
    none,
    f32_n16,
    f32_n32,
    f64_n8,
    f64_n16,
    f64_n32,
};

pub const BPackPath = enum {
    natural,
    dynamic,
    transpose4,
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
    amx: AmxKernel = .none,
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
        beta: T,
        c: [*]T,
        ldc: BlasInt,
        allow_sme: bool = false,
        kernel: catalog.KernelId = .auto,
        implementation: Implementation = .auto,
        execution: ExecutionPlan = .{},
    };
}

pub inline fn toUsize(x: BlasInt) usize {
    return @intCast(x);
}

pub inline fn matIndex(lda: BlasInt, row: usize, col: usize) usize {
    return row + col * toUsize(lda);
}
