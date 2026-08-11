// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Test-enumerable real GEMM implementation and evidence matrix.
//!
//! This module records coverage facts, not dispatch preference.  A target may
//! be implemented without having native performance evidence, and a rejected
//! experiment remains visible without becoming a selectable kernel.

const std = @import("std");

const catalog = @import("catalog.zig");

pub const Availability = enum {
    implemented,
    missing,
    rejected,
};

pub const Evidence = packed struct {
    build: bool = false,
    native_correctness: bool = false,
    native_performance: bool = false,
};

pub const Contract = struct {
    layouts: catalog.LayoutSupport,
    tails: catalog.TailSupport,
    epilogue: catalog.EpilogueSupport,
    pack_kind: catalog.PackKind,
    stack_workspace_bytes: usize,
    cache_workspace_bytes: usize,
    tile: catalog.Tile,
};

pub const Entry = struct {
    stable_id: []const u8,
    kernel: ?catalog.KernelId,
    scalar: catalog.ScalarKind,
    capability: catalog.IsaCapability,
    availability: Availability,
    lifecycle: catalog.Lifecycle,
    state: catalog.StateContract,
    evidence: Evidence,
    contract: ?Contract,
    evidence_note: []const u8,
};

fn contractForDescriptor(desc: catalog.Descriptor) Contract {
    return .{
        .layouts = desc.layouts,
        .tails = desc.tails,
        .epilogue = desc.epilogue,
        .pack_kind = desc.pack.kind,
        .stack_workspace_bytes = desc.pack.stack_bytes,
        .cache_workspace_bytes = desc.pack.cache_bytes,
        .tile = desc.tile,
    };
}

fn evidenceForKernel(kernel: catalog.KernelId) Evidence {
    return switch (kernel) {
        .generic_f32_4x4, .generic_f64_4x4 => .{
            .build = true,
            .native_correctness = true,
        },
        .aarch64_asimd_f32_12x8,
        .aarch64_asimd_f64_6x8,
        .aarch64_sme_f32_2mx2n,
        .aarch64_sme_f64_4mx2n,
        => .{
            .build = true,
            .native_correctness = true,
            .native_performance = true,
        },
        .aarch64_sve2_asimd_f32_12x8, .aarch64_sve2_asimd_f64_6x8 => .{
            .build = true,
            .native_correctness = true,
        },
        .x86_64_sse2_f32_packed,
        .x86_64_sse2_f64_packed,
        .x86_64_avx_f32_packed,
        .x86_64_avx_f64_packed,
        .x86_64_avx2_fma_f32_packed,
        .x86_64_avx2_fma_f64_packed,
        => .{ .build = true },
        .x86_64_avx512f_fma_f32_packed, .x86_64_avx512f_fma_f64_packed => .{
            .build = true,
            .native_correctness = true,
            .native_performance = true,
        },
        .auto => .{},
    };
}

fn noteForKernel(kernel: catalog.KernelId) []const u8 {
    return switch (kernel) {
        .generic_f32_4x4, .generic_f64_4x4 => "portable native correctness coverage; no kernel-specific throughput claim",
        .aarch64_asimd_f32_12x8, .aarch64_asimd_f64_6x8 => "ASIMD native correctness and focused/full-sweep performance records",
        .aarch64_sve2_asimd_f32_12x8, .aarch64_sve2_asimd_f64_6x8 => "SVE2-gated ASIMD wrapper; this is not a scalable-vector microkernel",
        .aarch64_sme_f32_2mx2n, .aarch64_sme_f64_4mx2n => "SME native correctness and focused/full-sweep performance records",
        .x86_64_sse2_f32_packed,
        .x86_64_sse2_f64_packed,
        .x86_64_avx_f32_packed,
        .x86_64_avx_f64_packed,
        .x86_64_avx2_fma_f32_packed,
        .x86_64_avx2_fma_f64_packed,
        => "build coverage only; native tier-specific forced-path evidence remains required",
        .x86_64_avx512f_fma_f32_packed, .x86_64_avx512f_fma_f64_packed => "AVX-512F/FMA native correctness and fresh-process performance records",
        .auto => "not registrable",
    };
}

fn descriptorEntry(desc: catalog.Descriptor) Entry {
    return .{
        .stable_id = desc.name,
        .kernel = desc.kernel,
        .scalar = desc.scalar,
        .capability = desc.capability,
        .availability = .implemented,
        .lifecycle = desc.lifecycle,
        .state = desc.state,
        .evidence = evidenceForKernel(desc.kernel),
        .contract = contractForDescriptor(desc),
        .evidence_note = noteForKernel(desc.kernel),
    };
}

fn amxContract(scalar: catalog.ScalarKind, register_m: usize, register_n: usize) Contract {
    return .{
        .layouts = .{ .transposed_b = true },
        .tails = .{
            .m = .whole_operation_fallback,
            .n = .whole_operation_fallback,
            .k = .native,
        },
        .epilogue = .{
            .arbitrary_alpha = false,
            .arbitrary_beta = false,
            .alpha_zero = false,
            .beta_zero = true,
        },
        .pack_kind = .b_panel_batched,
        .stack_workspace_bytes = if (scalar == .f32) 128 * 1024 else 256 * 1024,
        .cache_workspace_bytes = 8 * 1024 * 1024,
        .tile = .{
            .vector_lanes = if (scalar == .f32) 16 else 8,
            .register_m = register_m,
            .register_n = register_n,
            .n_panel = register_n,
            .k_unroll = 1,
        },
    };
}

fn amxEntry(stable_id: []const u8, scalar: catalog.ScalarKind, register_m: usize, register_n: usize) Entry {
    return .{
        .stable_id = stable_id,
        .kernel = null,
        .scalar = scalar,
        .capability = .apple_amx,
        .availability = .implemented,
        .lifecycle = .experimental,
        .state = catalog.stateContract(.apple_amx),
        .evidence = .{
            .build = true,
            .native_correctness = true,
            .native_performance = true,
        },
        .contract = amxContract(scalar, register_m, register_n),
        .evidence_note = "Private Apple ISA subkernel compiled out by default; -Dapple-amx=true requires a validated AArch64 macOS deployment, and retained native evidence applies only to that opt-in path",
    };
}

fn unavailableEntry(
    stable_id: []const u8,
    scalar: catalog.ScalarKind,
    capability: catalog.IsaCapability,
    availability: Availability,
    note: []const u8,
) Entry {
    return .{
        .stable_id = stable_id,
        .kernel = null,
        .scalar = scalar,
        .capability = capability,
        .availability = availability,
        .lifecycle = if (availability == .rejected) .rejected else .experimental,
        .state = switch (capability) {
            .aarch64_sme, .aarch64_sme2 => catalog.stateContract(.aarch64_streaming_za),
            .apple_amx => catalog.stateContract(.apple_amx),
            .x86_64_amx => catalog.stateContract(.x86_64_amx),
            else => .{},
        },
        .evidence = .{},
        .contract = null,
        .evidence_note = note,
    };
}

pub const entry_count = catalog.registered_descriptor_count + 5 + 8 + 2;

pub fn entries(streaming_vector_bytes: usize) [entry_count]Entry {
    const descriptors = catalog.registeredDescriptors(streaming_vector_bytes);
    var result: [entry_count]Entry = undefined;
    var next: usize = 0;
    for (descriptors) |desc| {
        result[next] = descriptorEntry(desc);
        next += 1;
    }

    result[next] = amxEntry("apple_amx_f32_n16", .f32, 16, 16);
    next += 1;
    result[next] = amxEntry("apple_amx_f32_n32", .f32, 32, 32);
    next += 1;
    result[next] = amxEntry("apple_amx_f64_n8", .f64, 8, 8);
    next += 1;
    result[next] = amxEntry("apple_amx_f64_n16", .f64, 32, 16);
    next += 1;
    result[next] = amxEntry("apple_amx_f64_n32", .f64, 16, 32);
    next += 1;

    result[next] = unavailableEntry("aarch64_sve_f32_scalable", .f32, .aarch64_sve, .missing, "no true scalable SVE real GEMM microkernel");
    next += 1;
    result[next] = unavailableEntry("aarch64_sve_f64_scalable", .f64, .aarch64_sve, .missing, "no true scalable SVE real GEMM microkernel");
    next += 1;
    result[next] = unavailableEntry("aarch64_sve2_f32_scalable", .f32, .aarch64_sve2, .missing, "current SVE2 ID delegates to the fixed-width ASIMD body");
    next += 1;
    result[next] = unavailableEntry("aarch64_sve2_f64_scalable", .f64, .aarch64_sve2, .missing, "current SVE2 ID delegates to the fixed-width ASIMD body");
    next += 1;
    result[next] = unavailableEntry("aarch64_sme2_f32", .f32, .aarch64_sme2, .missing, "no SME2-specific real GEMM microkernel");
    next += 1;
    result[next] = unavailableEntry("aarch64_sme2_f64", .f64, .aarch64_sme2, .missing, "no SME2-specific real GEMM microkernel");
    next += 1;
    result[next] = unavailableEntry("x86_64_amx_f32", .f32, .x86_64_amx, .missing, "no x86 AMX real GEMM microkernel");
    next += 1;
    result[next] = unavailableEntry("x86_64_amx_f64", .f64, .x86_64_amx, .missing, "no x86 AMX real GEMM microkernel");
    next += 1;

    result[next] = unavailableEntry("aarch64_sme_f32_grouped_load_2x2", .f32, .aarch64_sme, .rejected, "focused rerun regressed the retained SME panel and the experiment was reverted");
    next += 1;
    result[next] = unavailableEntry("apple_amx_f32_n32_kchunk", .f32, .apple_amx, .rejected, "256-wide K chunking failed to recover high-K throughput and was reverted");
    next += 1;

    std.debug.assert(next == result.len);
    return result;
}

test "coverage report names every registered real GEMM kernel exactly once" {
    const report = entries(64);
    for (std.meta.tags(catalog.KernelId)) |kernel| {
        if (kernel == .auto) continue;
        var matches: usize = 0;
        for (report) |entry| {
            if (entry.kernel == kernel) matches += 1;
        }
        try std.testing.expectEqual(@as(usize, 1), matches);
    }
}

test "coverage report has stable unique ids and monotonic evidence" {
    const report = entries(64);
    for (report, 0..) |entry, i| {
        try std.testing.expect(entry.stable_id.len != 0);
        if (entry.evidence.native_performance) try std.testing.expect(entry.evidence.native_correctness);
        if (entry.evidence.native_correctness) try std.testing.expect(entry.evidence.build);
        if (entry.availability != .implemented) try std.testing.expect(entry.kernel == null);
        for (report[i + 1 ..]) |other| {
            try std.testing.expect(!std.mem.eql(u8, entry.stable_id, other.stable_id));
        }
    }
}

test "coverage report exposes implemented experimental rejected and missing cells" {
    const report = entries(64);
    var implemented: usize = 0;
    var experimental: usize = 0;
    var rejected: usize = 0;
    var missing: usize = 0;
    for (report) |entry| {
        if (entry.availability == .implemented) implemented += 1;
        if (entry.lifecycle == .experimental and entry.availability == .implemented) experimental += 1;
        if (entry.availability == .rejected) rejected += 1;
        if (entry.availability == .missing) missing += 1;
    }
    try std.testing.expect(implemented != 0);
    try std.testing.expect(experimental != 0);
    try std.testing.expect(rejected != 0);
    try std.testing.expect(missing != 0);
}

test "coverage exposes state ownership for SME Apple AMX and missing x86 AMX" {
    const report = entries(64);
    var saw_sme = false;
    var saw_apple_amx = false;
    var saw_x86_amx_gap = false;
    for (report) |entry| {
        switch (entry.capability) {
            .aarch64_sme, .aarch64_sme2 => {
                saw_sme = true;
                try std.testing.expectEqual(catalog.StateKind.aarch64_streaming_za, entry.state.kind);
                try std.testing.expect(entry.state.cleanup_on_success);
                try std.testing.expect(entry.state.cleanup_on_failure);
            },
            .apple_amx => {
                saw_apple_amx = true;
                try std.testing.expectEqual(catalog.StateKind.apple_amx, entry.state.kind);
                try std.testing.expect(entry.state.clobbers.matrix_accumulator);
            },
            .x86_64_amx => {
                saw_x86_amx_gap = true;
                try std.testing.expectEqual(Availability.missing, entry.availability);
                try std.testing.expectEqual(catalog.StateKind.x86_64_amx, entry.state.kind);
                try std.testing.expect(entry.state.os_permission_required);
                try std.testing.expect(entry.state.clobbers.tile_configuration);
            },
            else => {},
        }
    }
    try std.testing.expect(saw_sme);
    try std.testing.expect(saw_apple_amx);
    try std.testing.expect(saw_x86_amx_gap);
}
