// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Descriptor matching for no-transpose real GEMM.
//!
//! Kernel executable code lives in architecture modules. This file owns the
//! tunable scoring rules that combine shape, scalar epilogue requirements, pack
//! cost, and available descriptor metadata into a selected kernel.

const std = @import("std");

const catalog = @import("catalog.zig");
const gemm_task = @import("task.zig");

pub const Shape = struct {
    m: usize,
    n: usize,
    k: usize,
};

pub fn min3(a: usize, b: usize, c: usize) usize {
    return @min(a, @min(b, c));
}

pub fn max3(a: usize, b: usize, c: usize) usize {
    return @max(a, @max(b, c));
}

pub fn isSquareish(shape: Shape) bool {
    const min_dim = min3(shape.m, shape.n, shape.k);
    if (min_dim == 0) return false;
    return max3(shape.m, shape.n, shape.k) <= min_dim * 2;
}

pub fn isNarrowN(desc: catalog.Descriptor, n: usize) bool {
    return n <= desc.tile.n_panel * 4;
}

pub fn directKernelAllowed(comptime T: type, desc: catalog.Descriptor, shape: Shape, alpha: T, beta: T) bool {
    if (desc.family != .streaming_matrix) return false;
    if (alpha != 1 or beta != 0) return false;
    if (shape.m < desc.bounds.min_m_block or shape.n < desc.bounds.min_n_block or shape.k < desc.bounds.min_k_block) return false;
    return shape.m *| shape.n *| shape.k >= desc.bounds.min_work;
}

pub fn score(comptime T: type, desc: catalog.Descriptor, shape: Shape, alpha: T, beta: T, requested_threads: usize) i64 {
    const work = shape.m *| shape.n *| shape.k;
    const min_dim = min3(shape.m, shape.n, shape.k);
    const squareish = isSquareish(shape);
    var result: i64 = 0;

    switch (desc.family) {
        .generic => {
            result += 100;
            if (shape.m == 1 and shape.k >= 16) result += 900;
            if (shape.n == 1 and shape.k >= 16) result += 900;
            if (work <= 64 * 1024) result += 650;
            if (min_dim <= desc.tile.n_panel) result += 160;
            if (shape.m % desc.bounds.min_m_block != 0 or shape.n % desc.tile.n_panel != 0) result += 90;
        },
        .packed_simd => {
            if (shape.m == 1 and shape.k >= 16) result -= 700;
            if (shape.n == 1 and shape.k >= 16) result -= 700;
            if (work < desc.bounds.min_work / 2) result -= 180;
            result += 250 + @as(i64, @intCast(desc.tile.vector_lanes * desc.tile.register_n));
            if (shape.k >= desc.bounds.min_k_block) result += 120;
            if (shape.n >= desc.tile.n_panel) result += 100;
            if (shape.m >= desc.bounds.min_m_block) result += 80;
            if (work >= 1024 * 1024) result += 180;
            if (shape.m % desc.bounds.min_m_block == 0 and shape.n % desc.tile.n_panel == 0) result += 50;
        },
        .streaming_matrix => {
            if (!directKernelAllowed(T, desc, shape, alpha, beta)) return std.math.minInt(i64) / 2;
            result += 500;
            if (selectAmx(T, shape) != .none) result += 420;
            if (T == f32 and shape.m >= 512 and shape.n <= desc.tile.n_panel * 4 and shape.k >= 256) result += 380;
            if (T == f32 and shape.m >= 256 and shape.n >= 256 and shape.k >= 512) result += 420;
            if (squareish) result += 180;
            if (work >= 128 * 1024 * 1024) result += 260;
            if (requested_threads <= 1 and work <= 512 * 512 * 512) result += 120;
            if (shape.m < desc.tile.register_m or shape.n < desc.tile.register_n) result -= 200;
        },
    }

    const pack_elems = shape.k *| desc.tile.n_panel;
    if (desc.pack.kind != .none and desc.pack.stack_bytes != 0 and pack_elems * @sizeOf(T) <= desc.pack.stack_bytes) {
        result += 40;
    }
    if (T == f64 and desc.family == .streaming_matrix) result += 40;
    return result;
}

pub fn select(comptime T: type, candidates: catalog.CandidateList, shape: Shape, alpha: T, beta: T, requested_threads: usize) catalog.Descriptor {
    var best = candidates.at(0);
    var best_score = score(T, best, shape, alpha, beta, requested_threads);
    var index: usize = 1;
    while (index < candidates.len) : (index += 1) {
        const item = candidates.at(index);
        const item_score = score(T, item, shape, alpha, beta, requested_threads);
        if (item_score > best_score) {
            best = item;
            best_score = item_score;
        }
    }
    return best;
}

fn selectAmx(comptime T: type, shape: Shape) gemm_task.AmxKernel {
    if (shape.k == 0) return .none;
    if (T == f32) {
        if ((shape.m & 15) != 0 or (shape.n & 15) != 0) return .none;
        if (shape.k > 512) return .none;

        const short_wide = shape.m <= 64 and shape.n >= 512 and shape.k >= 128;
        const square = shape.m == shape.n and shape.k == shape.n and shape.m >= 64 and shape.m <= 768;
        const square512_chunk = shape.m == 512 and shape.k == 512 and shape.n <= 128;
        const square1024_chunk = shape.m == 1024 and shape.k == 1024 and shape.n <= 128;
        const narrow_n64_chunk = shape.m >= 1024 and shape.n == 32;
        const tall_panel = shape.m >= 128 and shape.n >= 32 and shape.n <= 128 and shape.k >= 256 and shape.k <= 1024 and !square512_chunk and !square1024_chunk;
        const high_k_small = shape.m == 128 and shape.n == 32 and shape.k >= 4096;
        const high_k_m512_n32 = shape.m == 512 and shape.n == 32 and shape.k == 2048;
        const low_k_large_n32 = shape.m >= 256 and shape.n >= 256 and shape.k <= 256;
        const tall_n16 = shape.m >= 512 and shape.n == 16 and shape.k >= 128 and shape.k <= 1024;
        const high_k_panel = shape.m >= 128 and shape.m <= 512 and shape.n >= 32 and shape.n <= 128 and shape.k >= 2048;
        if (!short_wide and !square and !tall_panel and !high_k_small and !high_k_m512_n32 and !high_k_panel and !low_k_large_n32 and !tall_n16) return .none;

        const square_n32 = square and (shape.m == 96 or shape.m == 128 or shape.m == 192 or shape.m == 256 or shape.m == 384 or shape.m == 512 or shape.m == 768);
        const short_wide_n32 = shape.m <= 64 and shape.n >= 512 and shape.k >= 128;
        const tall_panel_n32 = shape.m >= 128 and shape.n >= 32 and shape.n <= 128 and shape.k >= 128 and shape.k <= 1024 and !square512_chunk and !square1024_chunk and !narrow_n64_chunk;
        const high_k_chunk_n32 = shape.m == 128 and shape.n == 32 and shape.k >= 4096;
        const high_k_panel_n32 = high_k_panel;
        if ((shape.m & 31) == 0 and (shape.n & 31) == 0 and (low_k_large_n32 or square_n32 or short_wide_n32 or tall_panel_n32 or high_k_chunk_n32 or high_k_panel_n32 or narrow_n64_chunk)) {
            return .f32_n32;
        }
        return .f32_n16;
    }
    if (T == f64) {
        if ((shape.m & 7) != 0 or (shape.n & 7) != 0) return .none;
        const short_wide = shape.m <= 64 and shape.n >= 512 and shape.k >= 128;
        const square = shape.m == shape.n and shape.k == shape.n and shape.m >= 64 and shape.m <= 384;
        const high_k_panel = shape.m >= 128 and shape.m <= 512 and shape.n >= 32 and shape.n <= 128 and shape.k >= 2048;
        const tall_narrow_panel = shape.m >= 512 and shape.n >= 32 and shape.n <= 64 and shape.k >= 512 and shape.k <= 1024;
        const low_k_skinny_n32 = shape.m >= 2048 and shape.n >= 32 and shape.n <= 64 and shape.k >= 256 and shape.k <= 512;
        const low_k_large = shape.m >= 256 and shape.n >= 256 and shape.k <= 256;
        const mid_k_large = shape.m >= 256 and shape.n >= 256 and shape.k <= 1024;
        if (!short_wide and !square and !high_k_panel and !tall_narrow_panel and !low_k_skinny_n32 and !low_k_large and !mid_k_large) return .none;
        const square_large_n32 = square and shape.m >= 256;
        if (mid_k_large and shape.n <= 256 and (shape.m & 31) == 0 and (shape.n & 15) == 0) return .f64_n16;
        if ((short_wide or (shape.m == 64 or shape.m == 96) or square_large_n32 or high_k_panel or tall_narrow_panel or low_k_skinny_n32 or low_k_large or mid_k_large) and (shape.m & 15) == 0 and (shape.n & 31) == 0) return .f64_n32;
        if ((shape.m & 31) == 0 and (shape.n & 15) == 0) return .f64_n16;
        return .f64_n8;
    }
    return .none;
}

pub fn amxMBlock(amx: gemm_task.AmxKernel) usize {
    return switch (amx) {
        .f32_n16 => 16,
        .f32_n32 => 32,
        .f64_n8 => 8,
        .f64_n16 => 32,
        .f64_n32 => 16,
        .none => 0,
    };
}

pub fn amxNPanel(amx: gemm_task.AmxKernel) usize {
    return switch (amx) {
        .f32_n16 => 16,
        .f32_n32 => 32,
        .f64_n8 => 8,
        .f64_n16 => 16,
        .f64_n32 => 32,
        .none => 0,
    };
}

pub fn amxKernelCompatible(comptime T: type, amx: gemm_task.AmxKernel, shape: Shape) bool {
    if (shape.k == 0 or amx == .none) return false;
    const m_block = amxMBlock(amx);
    const n_panel = amxNPanel(amx);
    if (m_block == 0 or n_panel == 0) return false;
    if (shape.m % m_block != 0 or shape.n % n_panel != 0) return false;
    return switch (T) {
        f32 => shape.k <= 512 and (amx == .f32_n16 or amx == .f32_n32),
        f64 => amx == .f64_n8 or amx == .f64_n16 or amx == .f64_n32,
        else => false,
    };
}

fn selectF32AmxPartialN16(shape: Shape, amx: gemm_task.AmxKernel) bool {
    if (amx != .none) return false;
    if (shape.m % 16 != 0 or shape.k == 0 or shape.k > 512) return false;
    const n_full = shape.n - shape.n % 16;
    return n_full != 0 and n_full != shape.n;
}

fn selectBPack(comptime T: type, shape: Shape) gemm_task.BPackPath {
    if (T == f32) {
        if (shape.k % 4 != 0 or shape.n < 32) return .natural;
        const short_wide = shape.m <= 64 and shape.n >= 64 and shape.k >= 256;
        const tall_narrow = shape.n <= 64 and shape.m >= 512 and shape.k >= 256;
        const high_k_panel = shape.n <= 128 and shape.k >= 2048 and shape.m <= 512;
        const mid_k_wide = shape.m >= 128 and shape.n >= 256 and shape.k >= 512;
        return if (short_wide or tall_narrow or high_k_panel or mid_k_wide) .transpose4 else .natural;
    }
    if (T == f64) {
        return if (shape.m <= 64 and shape.n >= 64 and shape.k >= 256) .dynamic else .natural;
    }
    return .natural;
}

fn selectF32SmePanel(shape: Shape, tile: usize) gemm_task.SmeF32Panel {
    if (tile != 16) return .panels2x2;
    if (shape.k % 4 != 0) return .panels2x2;
    const panel2_cols = tile * 2;
    if (shape.n < panel2_cols or shape.n % panel2_cols != 0) return .panels2x2;

    const min_mn = @min(shape.m, shape.n);
    const max_mn = @max(shape.m, shape.n);
    const squareish = min_mn >= 96 and max_mn <= min_mn * 2 and shape.k >= 96 and shape.k <= 512;
    const high_k_squareish = min_mn >= 128 and max_mn <= min_mn * 2 and shape.k >= 768;
    const high_k_single_panel = shape.n == panel2_cols and shape.m >= 96 and shape.m <= 1024 and shape.k >= 1024;
    if (squareish or high_k_squareish or high_k_single_panel) return .panels2x2_u4;
    return .panels2x2;
}

fn selectSmePanelBatch(comptime T: type, desc: catalog.Descriptor, shape: Shape, requested_threads: usize, performance_l2_bytes: usize) usize {
    if (desc.family != .streaming_matrix) return 1;
    if (T != f32 and T != f64) return 1;
    const tile = desc.tile.vector_lanes;
    const panel_cols = tile * 2;
    if (panel_cols == 0) return 1;
    const panels = shape.n / panel_cols;
    if (panels <= 1) return 1;

    const max_batch: usize = 8;
    const per_panel_bytes = shape.k *| panel_cols *| @sizeOf(T);
    if (per_panel_bytes == 0) return 1;
    const threads = @max(@as(usize, 1), requested_threads);
    const l2_bytes = if (performance_l2_bytes != 0) performance_l2_bytes else 16 * 1024 * 1024;
    const l2_budget = l2_bytes / (threads * 2);
    const pack_budget = @max(@as(usize, 256 * 1024), l2_budget);
    const cache_batch = @max(@as(usize, 1), pack_budget / per_panel_bytes);
    return @min(panels, @min(max_batch, cache_batch));
}

fn selectPackWorkspace(desc: catalog.Descriptor) gemm_task.PackWorkspacePlan {
    return .{
        .stack_bytes = desc.pack.stack_bytes,
        .cache_bytes = switch (desc.family) {
            .streaming_matrix => 16 * 1024 * 1024,
            else => 0,
        },
    };
}

fn selectAmxPackWorkspace(comptime T: type, amx: gemm_task.AmxKernel) gemm_task.PackWorkspacePlan {
    if (amx == .none) return .{};
    return .{
        .stack_bytes = if (T == f32) 128 * 1024 else if (T == f64) 256 * 1024 else 0,
        .cache_bytes = 8 * 1024 * 1024,
    };
}

pub fn executionPlan(comptime T: type, desc: catalog.Descriptor, shape: Shape, requested_threads: usize, performance_l2_bytes: usize) gemm_task.ExecutionPlan {
    var result: gemm_task.ExecutionPlan = .{};
    result.pack = selectPackWorkspace(desc);
    if (desc.family != .streaming_matrix) return result;

    result.amx = selectAmx(T, shape);
    result.amx_partial_n16 = T == f32 and selectF32AmxPartialN16(shape, result.amx);
    result.amx_pack = selectAmxPackWorkspace(T, result.amx);
    if (result.amx_partial_n16) result.amx_pack = selectAmxPackWorkspace(T, .f32_n16);
    result.b_pack = selectBPack(T, shape);
    if (T == f32) result.f32_panel = selectF32SmePanel(shape, desc.tile.vector_lanes);
    result.sme_panel_batch = selectSmePanelBatch(T, desc, shape, requested_threads, performance_l2_bytes);
    return result;
}
