// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Static CPU GEMM kernel descriptors.
//!
//! The executable kernels stay in the architecture-specific modules.  This file
//! is the metadata catalog consumed by the GEMM planner so dispatch policy can be
//! expressed in terms of tile, pack, unroll, ISA, and minimum useful work.

const std = @import("std");
const contract = @import("../../contract.zig");
const packed_params = @import("packed_params.zig");

pub const structured = @import("structured_catalog.zig");

pub const OperationKind = contract.OperationKind;
pub const ScalarKind = contract.ScalarKind;
pub const IsaCapability = contract.IsaCapability;
pub const Lifecycle = contract.Lifecycle;
pub const TailStrategy = contract.TailStrategy;
pub const TailSupport = contract.TailSupport;
pub const LayoutSupport = contract.LayoutSupport;
pub const EpilogueSupport = contract.EpilogueSupport;
pub const StateKind = contract.StateKind;
pub const StateContract = contract.StateContract;
pub const stateContract = contract.stateContract;
pub const FallbackSemantics = contract.FallbackSemantics;

pub fn contractScalarKind(comptime T: type) ScalarKind {
    return contract.scalarKind(T);
}

pub const KernelFamily = enum {
    generic,
    packed_simd,
    streaming_matrix,
};

pub const KernelId = enum {
    auto,
    generic_f32_4x4,
    generic_f64_4x4,
    aarch64_asimd_f32_12x8,
    aarch64_asimd_f64_6x8,
    aarch64_sve2_asimd_f32_12x8,
    aarch64_sve2_asimd_f64_6x8,
    aarch64_sme_f32_2mx2n,
    aarch64_sme_f64_4mx2n,
    x86_64_sse2_f32_packed,
    x86_64_sse2_f64_packed,
    x86_64_avx_f32_packed,
    x86_64_avx_f64_packed,
    x86_64_avx2_fma_f32_packed,
    x86_64_avx2_fma_f64_packed,
    x86_64_avx512f_fma_f32_packed,
    x86_64_avx512f_fma_f64_packed,
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
    cache_bytes: usize,
};

pub const Descriptor = struct {
    name: []const u8,
    kernel: KernelId,
    fallback: KernelId,
    operation: OperationKind,
    scalar: ScalarKind,
    capability: IsaCapability,
    lifecycle: Lifecycle,
    family: KernelFamily,
    layouts: LayoutSupport,
    tails: TailSupport,
    epilogue: EpilogueSupport,
    state: StateContract,
    fallback_semantics: FallbackSemantics,
    tile: Tile,
    bounds: Bounds,
    pack: Packing,
};

pub const ComplexKernelFamily = enum {
    portable,
    compact,
    three_m,
    expanded_real,
    vector_edge,
};

pub const ComplexKernelId = enum {
    portable_c32,
    portable_c64,
    compact_c32,
    compact_c64,
    three_m_c32,
    three_m_c64,
    expanded_real_c32,
    expanded_real_c64,
    vector_edge_c32,
    vector_edge_c64,
};

pub const ComplexMaterialization = enum {
    none,
    planar_three_m,
    expanded_real_matrix,
};

pub const ComplexWorkspaceFormula = enum {
    none,
    three_a_three_b_three_c,
    expanded_a_b_c,
};

pub const ComplexPacking = enum {
    none,
    materialized_planar_real_inputs,
    materialized_expanded_real_inputs,
};

pub const ComplexCombine = enum {
    direct_complex_epilogue,
    three_product_combine,
    expanded_real_scatter,
    gemv_delegation,
};

pub const ConjugationMode = enum {
    not_supported,
    indexed_load,
    materialized,
};

pub const ComplexDescriptor = struct {
    name: []const u8,
    kernel: ComplexKernelId,
    fallback: ?ComplexKernelId,
    scalar: ScalarKind,
    lifecycle: Lifecycle,
    family: ComplexKernelFamily,
    layouts: LayoutSupport,
    epilogue: EpilogueSupport,
    tails: TailSupport,
    conjugation: ConjugationMode,
    materialization: ComplexMaterialization,
    packing: ComplexPacking,
    workspace_formula: ComplexWorkspaceFormula,
    max_cached_workspace_bytes: usize,
    combine: ComplexCombine,
    output: contract.OutputOwnership,
    fallback_semantics: FallbackSemantics,
};

pub const complex_registry = buildComplexRegistry();

fn buildComplexRegistry() [10]ComplexDescriptor {
    const result = [_]ComplexDescriptor{
        makeComplexDescriptor(.complex_f32, .portable),
        makeComplexDescriptor(.complex_f64, .portable),
        makeComplexDescriptor(.complex_f32, .compact),
        makeComplexDescriptor(.complex_f64, .compact),
        makeComplexDescriptor(.complex_f32, .three_m),
        makeComplexDescriptor(.complex_f64, .three_m),
        makeComplexDescriptor(.complex_f32, .expanded_real),
        makeComplexDescriptor(.complex_f64, .expanded_real),
        makeComplexDescriptor(.complex_f32, .vector_edge),
        makeComplexDescriptor(.complex_f64, .vector_edge),
    };
    validateComplexRegistry(result);
    return result;
}

fn makeComplexDescriptor(comptime scalar: ScalarKind, comptime family: ComplexKernelFamily) ComplexDescriptor {
    const portable = family == .portable;
    const materialized = family == .three_m or family == .expanded_real;
    const all_layouts = LayoutSupport{
        .transposed_a = true,
        .transposed_b = true,
        .conjugated_a = true,
        .conjugated_b = true,
    };
    const layouts = switch (family) {
        .portable, .three_m => all_layouts,
        .expanded_real => if (scalar == .complex_f32) all_layouts else LayoutSupport{},
        .compact, .vector_edge => LayoutSupport{},
    };
    const restricted_epilogue = EpilogueSupport{
        .arbitrary_alpha = false,
        .arbitrary_beta = false,
        .alpha_zero = false,
        .beta_zero = true,
    };
    return .{
        .name = std.fmt.comptimePrint("gemm.{s}.{s}", .{ @tagName(scalar), @tagName(family) }),
        .kernel = complexKernelId(scalar, family),
        .fallback = complexFallback(scalar, family),
        .scalar = scalar,
        .lifecycle = if (portable) .portable_fallback else .production,
        .family = family,
        .layouts = layouts,
        .epilogue = if (materialized) restricted_epilogue else .{},
        .tails = .{
            .m = if (family == .compact) .scalar_cleanup else .native,
            .n = if (family == .compact) .scalar_cleanup else .native,
            .k = .native,
        },
        .conjugation = if (materialized)
            .materialized
        else if (portable)
            .indexed_load
        else
            .not_supported,
        .materialization = switch (family) {
            .three_m => .planar_three_m,
            .expanded_real => .expanded_real_matrix,
            else => .none,
        },
        .workspace_formula = switch (family) {
            .three_m => .three_a_three_b_three_c,
            .expanded_real => .expanded_a_b_c,
            else => .none,
        },
        .packing = switch (family) {
            .three_m => .materialized_planar_real_inputs,
            .expanded_real => .materialized_expanded_real_inputs,
            else => .none,
        },
        .max_cached_workspace_bytes = if (materialized) 64 * 1024 * 1024 else 0,
        .combine = switch (family) {
            .portable, .compact => .direct_complex_epilogue,
            .three_m => .three_product_combine,
            .expanded_real => .expanded_real_scatter,
            .vector_edge => .gemv_delegation,
        },
        .output = .full_general_matrix,
        .fallback_semantics = if (portable) .terminal_portable else .whole_operation,
    };
}

fn complexKernelId(scalar: ScalarKind, family: ComplexKernelFamily) ComplexKernelId {
    return switch (scalar) {
        .complex_f32 => switch (family) {
            .portable => .portable_c32,
            .compact => .compact_c32,
            .three_m => .three_m_c32,
            .expanded_real => .expanded_real_c32,
            .vector_edge => .vector_edge_c32,
        },
        .complex_f64 => switch (family) {
            .portable => .portable_c64,
            .compact => .compact_c64,
            .three_m => .three_m_c64,
            .expanded_real => .expanded_real_c64,
            .vector_edge => .vector_edge_c64,
        },
        else => unreachable,
    };
}

fn complexFallback(scalar: ScalarKind, family: ComplexKernelFamily) ?ComplexKernelId {
    return switch (family) {
        .portable => null,
        .compact => complexKernelId(scalar, .portable),
        .three_m => complexKernelId(scalar, .compact),
        .expanded_real => complexKernelId(scalar, .three_m),
        .vector_edge => complexKernelId(scalar, .compact),
    };
}

fn validateComplexRegistry(items: [10]ComplexDescriptor) void {
    for (items, 0..) |descriptor, index| {
        if (descriptor.scalar != .complex_f32 and descriptor.scalar != .complex_f64) {
            @compileError("complex GEMM descriptor has a real scalar kind");
        }
        if (descriptor.name.len == 0) @compileError("complex GEMM descriptor name is empty");
        if (descriptor.family == .portable) {
            if (descriptor.fallback != null or descriptor.lifecycle != .portable_fallback or
                descriptor.fallback_semantics != .terminal_portable)
            {
                @compileError("portable complex GEMM descriptor is not terminal");
            }
        } else {
            if (descriptor.fallback == null or descriptor.fallback_semantics != .whole_operation) {
                @compileError("optimized complex GEMM descriptor lacks whole-operation fallback");
            }
            if (!containsComplexKernel(items, descriptor.fallback.?)) {
                @compileError("complex GEMM fallback is absent from registry");
            }
        }
        const has_workspace = descriptor.workspace_formula != .none;
        if (has_workspace != (descriptor.materialization != .none) or
            has_workspace != (descriptor.packing != .none) or
            has_workspace != (descriptor.max_cached_workspace_bytes != 0))
        {
            @compileError("complex GEMM materialization and workspace contracts disagree");
        }
        if ((descriptor.layouts.conjugated_a or descriptor.layouts.conjugated_b) and
            descriptor.conjugation == .not_supported)
        {
            @compileError("complex GEMM conjugated layout lacks a conjugation contract");
        }
        for (items[index + 1 ..]) |other| {
            if (descriptor.kernel == other.kernel) @compileError("duplicate complex GEMM kernel id");
            if (equalString(descriptor.name, other.name)) @compileError("duplicate complex GEMM descriptor name");
        }
    }
}

fn containsComplexKernel(items: [10]ComplexDescriptor, kernel: ComplexKernelId) bool {
    for (items) |descriptor| if (descriptor.kernel == kernel) return true;
    return false;
}

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

pub fn validatedCandidateList(comptime descriptors: anytype) CandidateList {
    comptime validateDescriptorSet(descriptors);
    return candidateList(descriptors);
}

fn validateDescriptorSet(comptime descriptors: anytype) void {
    comptime {
        for (descriptors, 0..) |desc, i| {
            validateDescriptor(desc);
            for (descriptors, 0..) |other, j| {
                if (j <= i) continue;
                if (desc.kernel == other.kernel) @compileError("duplicate GEMM kernel id in candidate list");
                if (equalString(desc.name, other.name)) @compileError("duplicate GEMM kernel name in candidate list");
            }
            if (desc.fallback != desc.kernel and !containsKernel(descriptors, desc.fallback)) {
                @compileError("GEMM candidate fallback is absent from candidate list");
            }
        }
    }
}

fn equalString(a: []const u8, b: []const u8) bool {
    if (a.len != b.len) return false;
    for (a, b) |lhs, rhs| if (lhs != rhs) return false;
    return true;
}

fn containsKernel(descriptors: anytype, kernel: KernelId) bool {
    for (descriptors) |desc| if (desc.kernel == kernel) return true;
    return false;
}

fn validateDescriptor(desc: Descriptor) void {
    if (desc.name.len == 0) @compileError("GEMM kernel name must not be empty");
    if (desc.kernel == .auto) @compileError("auto is not a registrable GEMM kernel id");
    if (desc.operation != .matrix_matrix) @compileError("GEMM descriptor has the wrong operation kind");
    if (desc.tile.vector_lanes == 0 or desc.tile.register_m == 0 or desc.tile.register_n == 0 or desc.tile.n_panel == 0 or desc.tile.k_unroll == 0) {
        @compileError("GEMM descriptor tile fields must be nonzero");
    }
    if (desc.bounds.min_m_block == 0 or desc.bounds.min_n_block == 0 or desc.bounds.min_k_block == 0) {
        @compileError("GEMM descriptor block bounds must be nonzero");
    }
    if (desc.pack.kind == .none and (desc.pack.stack_bytes != 0 or desc.pack.cache_bytes != 0)) {
        @compileError("unpacked GEMM descriptor reserves pack workspace");
    }
    if (desc.pack.kind != .none and desc.pack.stack_bytes == 0) @compileError("packed GEMM descriptor lacks a stack workspace bound");
    if (desc.family == .streaming_matrix and desc.pack.cache_bytes == 0) {
        @compileError("streaming GEMM descriptor lacks a cache workspace bound");
    }
    if (desc.family != .streaming_matrix and desc.pack.cache_bytes != 0) {
        @compileError("non-streaming GEMM descriptor reserves cache workspace");
    }
    if (desc.family == .streaming_matrix and desc.state.kind != .aarch64_streaming_za) {
        @compileError("streaming GEMM descriptor lacks its ZA state contract");
    }
    if (desc.family != .streaming_matrix and desc.state.kind != .none) {
        @compileError("non-streaming GEMM descriptor declares streaming state");
    }
    contract.validateStateContract(desc.state);
    if (desc.lifecycle.defaultEligible() and !hasExecutorMapping(desc.kernel)) {
        @compileError("default-eligible GEMM descriptor lacks an executor mapping");
    }
}

pub fn hasExecutorMapping(kernel: KernelId) bool {
    return switch (kernel) {
        .auto => false,
        .generic_f32_4x4,
        .generic_f64_4x4,
        .aarch64_asimd_f32_12x8,
        .aarch64_asimd_f64_6x8,
        .aarch64_sve2_asimd_f32_12x8,
        .aarch64_sve2_asimd_f64_6x8,
        .aarch64_sme_f32_2mx2n,
        .aarch64_sme_f64_4mx2n,
        .x86_64_sse2_f32_packed,
        .x86_64_sse2_f64_packed,
        .x86_64_avx_f32_packed,
        .x86_64_avx_f64_packed,
        .x86_64_avx2_fma_f32_packed,
        .x86_64_avx2_fma_f64_packed,
        .x86_64_avx512f_fma_f32_packed,
        .x86_64_avx512f_fma_f64_packed,
        => true,
    };
}

pub fn genericKernelId(comptime T: type) KernelId {
    if (T == f32) return .generic_f32_4x4;
    if (T == f64) return .generic_f64_4x4;
    @compileError("GEMM catalog supports f32 and f64");
}

fn makeDescriptor(
    comptime T: type,
    name: []const u8,
    kernel: KernelId,
    family: KernelFamily,
    capability: IsaCapability,
    tile: Tile,
    bounds: Bounds,
    pack: Packing,
) Descriptor {
    const is_generic = family == .generic;
    const is_streaming = family == .streaming_matrix;
    return .{
        .name = name,
        .kernel = kernel,
        .fallback = if (is_generic) kernel else genericKernelId(T),
        .operation = .matrix_matrix,
        .scalar = contract.scalarKind(T),
        .capability = capability,
        .lifecycle = if (is_generic) .portable_fallback else .production,
        .family = family,
        .layouts = .{ .transposed_b = family == .packed_simd or (is_streaming and T == f32) },
        .tails = if (is_generic) .{
            .m = .native,
            .n = .native,
            .k = .native,
        } else if (is_streaming) .{
            .m = .scalar_cleanup,
            .n = .scalar_cleanup,
            .k = .native,
        } else .{
            .m = .scalar_cleanup,
            .n = .scalar_cleanup,
            .k = .native,
        },
        .epilogue = if (is_streaming) .{
            .arbitrary_alpha = false,
            .arbitrary_beta = false,
            .alpha_zero = false,
            .beta_zero = true,
        } else .{},
        .state = if (is_streaming) contract.stateContract(.aarch64_streaming_za) else .{},
        .fallback_semantics = if (is_generic) .terminal_portable else .whole_operation,
        .tile = tile,
        .bounds = bounds,
        .pack = pack,
    };
}

fn packedSimdDescriptor(
    comptime T: type,
    comptime name: []const u8,
    comptime kernel: KernelId,
    comptime capability: IsaCapability,
    comptime shape: packed_params.PackedSimdShape,
) Descriptor {
    return makeDescriptor(
        T,
        name,
        kernel,
        .packed_simd,
        capability,
        .{
            .vector_lanes = shape.lane_count,
            .register_m = shape.lane_count * shape.row_groups,
            .register_n = shape.tile_n,
            .n_panel = shape.tile_n,
            .k_unroll = shape.k_unroll,
        },
        .{
            .min_m_block = shape.lane_count,
            .min_n_block = shape.tile_n,
            .min_k_block = 16,
            .min_work = 64 * 1024,
        },
        .{ .kind = .b_panel, .stack_bytes = shape.max_stack_pack_bytes, .cache_bytes = 0 },
    );
}

pub fn genericDescriptor(comptime T: type) Descriptor {
    if (T == f32) {
        return makeDescriptor(
            T,
            "generic_f32_4x4",
            .generic_f32_4x4,
            .generic,
            .generic,
            .{ .vector_lanes = 4, .register_m = 4, .register_n = 4, .n_panel = 4, .k_unroll = 1 },
            .{ .min_m_block = 4, .min_n_block = 4, .min_k_block = 1, .min_work = 0 },
            .{ .kind = .none, .stack_bytes = 0, .cache_bytes = 0 },
        );
    }
    if (T == f64) {
        return makeDescriptor(
            T,
            "generic_f64_4x4",
            .generic_f64_4x4,
            .generic,
            .generic,
            .{ .vector_lanes = 2, .register_m = 4, .register_n = 4, .n_panel = 4, .k_unroll = 1 },
            .{ .min_m_block = 2, .min_n_block = 4, .min_k_block = 1, .min_work = 0 },
            .{ .kind = .none, .stack_bytes = 0, .cache_bytes = 0 },
        );
    }
    @compileError("GEMM catalog supports f32 and f64");
}

pub fn aarch64AsimdDescriptor(comptime T: type) Descriptor {
    const shape = packed_params.aarch64AsimdShape(T);
    if (T == f32) {
        return packedSimdDescriptor(
            T,
            "aarch64_asimd_f32_12x8",
            .aarch64_asimd_f32_12x8,
            .aarch64_asimd_fma,
            shape,
        );
    }
    if (T == f64) {
        return packedSimdDescriptor(
            T,
            "aarch64_asimd_f64_6x8",
            .aarch64_asimd_f64_6x8,
            .aarch64_asimd_fma,
            shape,
        );
    }
    @compileError("GEMM catalog supports f32 and f64");
}

pub fn aarch64Sve2Descriptor(comptime T: type) Descriptor {
    var desc = aarch64AsimdDescriptor(T);
    desc.name = if (T == f32) "aarch64_sve2_asimd_f32_12x8" else "aarch64_sve2_asimd_f64_6x8";
    desc.kernel = if (T == f32) .aarch64_sve2_asimd_f32_12x8 else .aarch64_sve2_asimd_f64_6x8;
    desc.capability = .aarch64_sve2;
    return desc;
}

pub fn aarch64SmeDescriptor(comptime T: type, streaming_vector_bytes: usize) Descriptor {
    const svl = if (streaming_vector_bytes == 0) 16 else streaming_vector_bytes;
    if (T == f32) {
        const tile: usize = @max(@as(usize, 4), svl / @sizeOf(f32));
        return makeDescriptor(
            T,
            "aarch64_sme_f32_2mx2n",
            .aarch64_sme_f32_2mx2n,
            .streaming_matrix,
            .aarch64_sme,
            .{ .vector_lanes = tile, .register_m = tile * 2, .register_n = tile * 2, .n_panel = tile * 2, .k_unroll = 4 },
            .{ .min_m_block = tile, .min_n_block = tile, .min_k_block = 32, .min_work = 128 * 1024 },
            .{ .kind = .b_panel_batched, .stack_bytes = 256 * 1024, .cache_bytes = 16 * 1024 * 1024 },
        );
    }
    if (T == f64) {
        const tile: usize = @max(@as(usize, 2), svl / @sizeOf(f64));
        return makeDescriptor(
            T,
            "aarch64_sme_f64_4mx2n",
            .aarch64_sme_f64_4mx2n,
            .streaming_matrix,
            .aarch64_sme,
            .{ .vector_lanes = tile, .register_m = tile * 4, .register_n = tile * 2, .n_panel = tile * 2, .k_unroll = 4 },
            .{ .min_m_block = tile, .min_n_block = tile, .min_k_block = 32, .min_work = 128 * 1024 },
            .{ .kind = .b_panel_batched, .stack_bytes = 256 * 1024, .cache_bytes = 16 * 1024 * 1024 },
        );
    }
    @compileError("GEMM catalog supports f32 and f64");
}

pub fn x86Descriptor(comptime T: type, comptime capability: IsaCapability) Descriptor {
    if (comptime capability != .x86_64_sse2 and
        capability != .x86_64_avx and
        capability != .x86_64_avx2_fma and
        capability != .x86_64_avx512f_fma)
    {
        @compileError("unsupported x86 GEMM capability");
    }
    const is_avx = capability != .x86_64_sse2;
    const is_avx512 = capability == .x86_64_avx512f_fma;
    const shape = packed_params.x86Shape(T, is_avx, is_avx512);
    if (T == f32) {
        const kernel: KernelId = switch (capability) {
            .x86_64_avx512f_fma => .x86_64_avx512f_fma_f32_packed,
            .x86_64_avx2_fma => .x86_64_avx2_fma_f32_packed,
            .x86_64_avx => .x86_64_avx_f32_packed,
            .x86_64_sse2 => .x86_64_sse2_f32_packed,
            else => unreachable,
        };
        return packedSimdDescriptor(
            T,
            switch (capability) {
                .x86_64_avx512f_fma => "x86_64_avx512f_fma_f32_packed",
                .x86_64_avx2_fma => "x86_64_avx2_fma_f32_packed",
                .x86_64_avx => "x86_64_avx_f32_packed",
                .x86_64_sse2 => "x86_64_sse2_f32_packed",
                else => unreachable,
            },
            kernel,
            capability,
            shape,
        );
    }
    if (T == f64) {
        const kernel: KernelId = switch (capability) {
            .x86_64_avx512f_fma => .x86_64_avx512f_fma_f64_packed,
            .x86_64_avx2_fma => .x86_64_avx2_fma_f64_packed,
            .x86_64_avx => .x86_64_avx_f64_packed,
            .x86_64_sse2 => .x86_64_sse2_f64_packed,
            else => unreachable,
        };
        return packedSimdDescriptor(
            T,
            switch (capability) {
                .x86_64_avx512f_fma => "x86_64_avx512f_fma_f64_packed",
                .x86_64_avx2_fma => "x86_64_avx2_fma_f64_packed",
                .x86_64_avx => "x86_64_avx_f64_packed",
                .x86_64_sse2 => "x86_64_sse2_f64_packed",
                else => unreachable,
            },
            kernel,
            capability,
            shape,
        );
    }
    @compileError("GEMM catalog supports f32 and f64");
}

pub const registered_descriptor_count = 16;

pub fn registeredDescriptors(streaming_vector_bytes: usize) [registered_descriptor_count]Descriptor {
    return .{
        genericDescriptor(f32),
        genericDescriptor(f64),
        aarch64AsimdDescriptor(f32),
        aarch64AsimdDescriptor(f64),
        aarch64Sve2Descriptor(f32),
        aarch64Sve2Descriptor(f64),
        aarch64SmeDescriptor(f32, streaming_vector_bytes),
        aarch64SmeDescriptor(f64, streaming_vector_bytes),
        x86Descriptor(f32, .x86_64_sse2),
        x86Descriptor(f64, .x86_64_sse2),
        x86Descriptor(f32, .x86_64_avx),
        x86Descriptor(f64, .x86_64_avx),
        x86Descriptor(f32, .x86_64_avx2_fma),
        x86Descriptor(f64, .x86_64_avx2_fma),
        x86Descriptor(f32, .x86_64_avx512f_fma),
        x86Descriptor(f64, .x86_64_avx512f_fma),
    };
}

pub fn descriptorForKernel(kernel: KernelId, streaming_vector_bytes: usize) ?Descriptor {
    const descriptors = registeredDescriptors(streaming_vector_bytes);
    for (descriptors) |desc| {
        if (desc.kernel == kernel) return desc;
    }
    return null;
}

test "GEMM candidate registries satisfy the common contract" {
    _ = validatedCandidateList(.{genericDescriptor(f32)});
    _ = validatedCandidateList(.{genericDescriptor(f64)});
    _ = validatedCandidateList(.{ aarch64AsimdDescriptor(f32), genericDescriptor(f32) });
    _ = validatedCandidateList(.{ aarch64AsimdDescriptor(f64), genericDescriptor(f64) });
    _ = validatedCandidateList(.{ aarch64Sve2Descriptor(f32), aarch64AsimdDescriptor(f32), genericDescriptor(f32) });
    _ = validatedCandidateList(.{ aarch64Sve2Descriptor(f64), aarch64AsimdDescriptor(f64), genericDescriptor(f64) });
    _ = validatedCandidateList(.{ aarch64SmeDescriptor(f32, 64), aarch64Sve2Descriptor(f32), genericDescriptor(f32) });
    _ = validatedCandidateList(.{ aarch64SmeDescriptor(f64, 64), aarch64AsimdDescriptor(f64), genericDescriptor(f64) });
    _ = validatedCandidateList(.{ x86Descriptor(f32, .x86_64_sse2), genericDescriptor(f32) });
    _ = validatedCandidateList(.{ x86Descriptor(f64, .x86_64_avx), genericDescriptor(f64) });
    _ = validatedCandidateList(.{ x86Descriptor(f64, .x86_64_avx2_fma), genericDescriptor(f64) });
    _ = validatedCandidateList(.{ x86Descriptor(f32, .x86_64_avx512f_fma), genericDescriptor(f32) });
}

test "complete real GEMM registry satisfies descriptor and fallback contracts" {
    comptime validateDescriptorSet(registeredDescriptors(64));
}

test "complex GEMM registry declares materialization combine and fallback contracts" {
    try std.testing.expectEqual(@as(usize, 10), complex_registry.len);

    const expanded_c32 = complexDescriptorForKernel(.expanded_real_c32).?;
    try std.testing.expect(expanded_c32.layouts.transposed_a);
    try std.testing.expect(expanded_c32.layouts.conjugated_b);
    try std.testing.expectEqual(ConjugationMode.materialized, expanded_c32.conjugation);
    try std.testing.expectEqual(ComplexMaterialization.expanded_real_matrix, expanded_c32.materialization);
    try std.testing.expectEqual(ComplexPacking.materialized_expanded_real_inputs, expanded_c32.packing);
    try std.testing.expectEqual(ComplexCombine.expanded_real_scatter, expanded_c32.combine);
    try std.testing.expectEqual(ComplexKernelId.three_m_c32, expanded_c32.fallback.?);
    try std.testing.expect(!expanded_c32.epilogue.arbitrary_alpha);
    try std.testing.expect(expanded_c32.epilogue.beta_zero);

    const expanded_c64 = complexDescriptorForKernel(.expanded_real_c64).?;
    try std.testing.expect(!expanded_c64.layouts.transposed_a);
    try std.testing.expect(!expanded_c64.layouts.conjugated_b);

    const vector_edge = complexDescriptorForKernel(.vector_edge_c64).?;
    try std.testing.expectEqual(ComplexCombine.gemv_delegation, vector_edge.combine);
    try std.testing.expectEqual(ComplexKernelId.compact_c64, vector_edge.fallback.?);
    try std.testing.expectEqual(@as(usize, 0), vector_edge.max_cached_workspace_bytes);

    const portable = complexDescriptorForKernel(.portable_c32).?;
    try std.testing.expectEqual(Lifecycle.portable_fallback, portable.lifecycle);
    try std.testing.expectEqual(FallbackSemantics.terminal_portable, portable.fallback_semantics);
    try std.testing.expect(portable.fallback == null);
}

pub fn complexDescriptorForKernel(kernel: ComplexKernelId) ?ComplexDescriptor {
    for (complex_registry) |descriptor| {
        if (descriptor.kernel == kernel) return descriptor;
    }
    return null;
}
