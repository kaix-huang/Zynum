// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Static CPU GEMM kernel descriptors.
//!
//! The executable kernels stay in the architecture-specific modules.  This file
//! is the metadata catalog consumed by the GEMM planner so dispatch policy can be
//! expressed in terms of tile, pack, unroll, ISA, and minimum useful work.

pub const IsaLevel = enum {
    generic_vector,
    aarch64_asimd_fma,
    aarch64_sve2_asimd_fma,
    aarch64_sme_asimd_fma,
    x86_64_sse2,
    x86_64_avx,
    x86_64_avx2_fma,
    x86_64_avx512f_fma,
};

pub const KernelFamily = enum {
    generic,
    packed_simd,
    streaming_matrix,
};

pub const KernelId = enum {
    auto,
    generic_basic,
    aarch64_asimd_packed,
    aarch64_sve2_asimd_packed,
    aarch64_sme_streaming,
    x86_64_simd_packed,
};

pub const PackKind = enum {
    none,
    b_panel,
    b_panel_batched,
};

pub const Tile = struct {
    vector_lanes: usize,
    register_m: usize,
    register_n: usize,
    n_panel: usize,
    k_unroll: usize,
};

pub const Bounds = struct {
    min_m_block: usize,
    min_n_block: usize,
    min_k_block: usize,
    min_work: usize,
};

pub const Packing = struct {
    kind: PackKind,
    stack_bytes: usize,
};

pub const Descriptor = struct {
    name: []const u8,
    kernel: KernelId,
    family: KernelFamily,
    isa: IsaLevel,
    tile: Tile,
    bounds: Bounds,
    pack: Packing,
};

pub const max_candidates = 4;

pub const CandidateList = struct {
    len: usize,
    items: [max_candidates]Descriptor,

    pub fn at(self: CandidateList, index: usize) Descriptor {
        return self.items[index];
    }
};

pub fn candidateList(descriptors: anytype) CandidateList {
    comptime {
        if (descriptors.len == 0 or descriptors.len > max_candidates) {
            @compileError("invalid GEMM candidate count");
        }
    }

    var out: CandidateList = undefined;
    out.len = descriptors.len;
    inline for (descriptors, 0..) |desc, i| {
        out.items[i] = desc;
    }
    return out;
}

fn makeDescriptor(
    name: []const u8,
    kernel: KernelId,
    family: KernelFamily,
    isa: IsaLevel,
    tile: Tile,
    bounds: Bounds,
    pack: Packing,
) Descriptor {
    return .{
        .name = name,
        .kernel = kernel,
        .family = family,
        .isa = isa,
        .tile = tile,
        .bounds = bounds,
        .pack = pack,
    };
}

pub fn genericDescriptor(comptime T: type) Descriptor {
    if (T == f32) {
        return makeDescriptor(
            "generic_f32_4x4",
            .generic_basic,
            .generic,
            .generic_vector,
            .{ .vector_lanes = 4, .register_m = 4, .register_n = 4, .n_panel = 4, .k_unroll = 1 },
            .{ .min_m_block = 4, .min_n_block = 4, .min_k_block = 1, .min_work = 0 },
            .{ .kind = .none, .stack_bytes = 0 },
        );
    }
    if (T == f64) {
        return makeDescriptor(
            "generic_f64_4x4",
            .generic_basic,
            .generic,
            .generic_vector,
            .{ .vector_lanes = 2, .register_m = 4, .register_n = 4, .n_panel = 4, .k_unroll = 1 },
            .{ .min_m_block = 2, .min_n_block = 4, .min_k_block = 1, .min_work = 0 },
            .{ .kind = .none, .stack_bytes = 0 },
        );
    }
    @compileError("GEMM catalog supports f32 and f64");
}

pub fn aarch64AsimdDescriptor(comptime T: type) Descriptor {
    if (T == f32) {
        return makeDescriptor(
            "aarch64_asimd_f32_12x8",
            .aarch64_asimd_packed,
            .packed_simd,
            .aarch64_asimd_fma,
            .{ .vector_lanes = 4, .register_m = 12, .register_n = 8, .n_panel = 8, .k_unroll = 4 },
            .{ .min_m_block = 4, .min_n_block = 8, .min_k_block = 16, .min_work = 64 * 1024 },
            .{ .kind = .b_panel, .stack_bytes = 64 * 1024 },
        );
    }
    if (T == f64) {
        return makeDescriptor(
            "aarch64_asimd_f64_6x8",
            .aarch64_asimd_packed,
            .packed_simd,
            .aarch64_asimd_fma,
            .{ .vector_lanes = 2, .register_m = 6, .register_n = 8, .n_panel = 8, .k_unroll = 4 },
            .{ .min_m_block = 2, .min_n_block = 8, .min_k_block = 16, .min_work = 64 * 1024 },
            .{ .kind = .b_panel, .stack_bytes = 64 * 1024 },
        );
    }
    @compileError("GEMM catalog supports f32 and f64");
}

pub fn aarch64Sve2Descriptor(comptime T: type) Descriptor {
    var desc = aarch64AsimdDescriptor(T);
    desc.name = if (T == f32) "aarch64_sve2_asimd_f32_12x8" else "aarch64_sve2_asimd_f64_6x8";
    desc.kernel = .aarch64_sve2_asimd_packed;
    desc.isa = .aarch64_sve2_asimd_fma;
    return desc;
}

pub fn aarch64SmeDescriptor(comptime T: type, streaming_vector_bytes: usize) Descriptor {
    const svl = if (streaming_vector_bytes == 0) 16 else streaming_vector_bytes;
    if (T == f32) {
        const tile = @max(@as(usize, 4), svl / @sizeOf(f32));
        return makeDescriptor(
            "aarch64_sme_f32_2mx2n",
            .aarch64_sme_streaming,
            .streaming_matrix,
            .aarch64_sme_asimd_fma,
            .{ .vector_lanes = tile, .register_m = tile * 2, .register_n = tile * 2, .n_panel = tile * 2, .k_unroll = 4 },
            .{ .min_m_block = tile, .min_n_block = tile, .min_k_block = 32, .min_work = 128 * 1024 },
            .{ .kind = .b_panel_batched, .stack_bytes = 256 * 1024 },
        );
    }
    if (T == f64) {
        const tile = @max(@as(usize, 2), svl / @sizeOf(f64));
        return makeDescriptor(
            "aarch64_sme_f64_4mx2n",
            .aarch64_sme_streaming,
            .streaming_matrix,
            .aarch64_sme_asimd_fma,
            .{ .vector_lanes = tile, .register_m = tile * 4, .register_n = tile * 2, .n_panel = tile * 2, .k_unroll = 4 },
            .{ .min_m_block = tile, .min_n_block = tile, .min_k_block = 32, .min_work = 128 * 1024 },
            .{ .kind = .b_panel_batched, .stack_bytes = 256 * 1024 },
        );
    }
    @compileError("GEMM catalog supports f32 and f64");
}

pub fn x86Descriptor(comptime T: type, comptime isa: IsaLevel) Descriptor {
    const is_avx = isa == .x86_64_avx or isa == .x86_64_avx2_fma or isa == .x86_64_avx512f_fma;
    const is_avx512 = isa == .x86_64_avx512f_fma;
    if (T == f32) {
        return makeDescriptor(
            switch (isa) {
                .x86_64_avx512f_fma => "x86_64_avx512f_f32_packed",
                .x86_64_avx2_fma => "x86_64_avx2_f32_packed",
                .x86_64_avx => "x86_64_avx_f32_packed",
                else => "x86_64_sse2_f32_packed",
            },
            .x86_64_simd_packed,
            .packed_simd,
            isa,
            .{
                .vector_lanes = if (is_avx512) 16 else if (is_avx) 8 else 4,
                .register_m = if (is_avx512) 16 else if (is_avx) 8 else 4,
                .register_n = if (is_avx) 12 else 8,
                .n_panel = if (is_avx) 12 else 8,
                .k_unroll = 4,
            },
            .{
                .min_m_block = if (is_avx512) 16 else if (is_avx) 8 else 4,
                .min_n_block = if (is_avx) 12 else 8,
                .min_k_block = 16,
                .min_work = 64 * 1024,
            },
            .{ .kind = .b_panel, .stack_bytes = 64 * 1024 },
        );
    }
    if (T == f64) {
        return makeDescriptor(
            switch (isa) {
                .x86_64_avx512f_fma => "x86_64_avx512f_f64_packed",
                .x86_64_avx2_fma => "x86_64_avx2_f64_packed",
                .x86_64_avx => "x86_64_avx_f64_packed",
                else => "x86_64_sse2_f64_packed",
            },
            .x86_64_simd_packed,
            .packed_simd,
            isa,
            .{
                .vector_lanes = if (is_avx512) 8 else if (is_avx) 4 else 2,
                .register_m = if (is_avx512) 8 else if (is_avx) 4 else 2,
                .register_n = if (is_avx) 8 else 6,
                .n_panel = if (is_avx) 8 else 6,
                .k_unroll = 4,
            },
            .{
                .min_m_block = if (is_avx512) 8 else if (is_avx) 4 else 2,
                .min_n_block = if (is_avx) 8 else 6,
                .min_k_block = 16,
                .min_work = 64 * 1024,
            },
            .{ .kind = .b_panel, .stack_bytes = 64 * 1024 },
        );
    }
    @compileError("GEMM catalog supports f32 and f64");
}
