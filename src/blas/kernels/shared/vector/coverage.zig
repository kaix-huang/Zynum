// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Test-enumerable Level 1 architecture microkernel matrix.
//!
//! A row is an operation/type/ISA cell, not a performance promise. x86 rows
//! describe separately compiled target-tier specializations of the shared
//! fixed-SIMD source; they are not runtime force-tier entrypoints in one fat
//! binary. Missing cells stay visible until a distinct executable descriptor
//! is registered.

const std = @import("std");

const catalog = @import("catalog.zig");
const contract = @import("../../contract.zig");

pub const Availability = enum {
    implemented,
    missing,
    unsupported,
};

pub const Specialization = enum {
    architecture_entrypoint,
    build_tier,
};

pub const Evidence = packed struct {
    build: bool = false,
    native_correctness: bool = false,
    native_performance: bool = false,
};

pub const Entry = struct {
    stable_id: []const u8,
    kernel: ?catalog.KernelId,
    operation: catalog.VectorOperation,
    scalar: catalog.ScalarKind,
    capability: catalog.IsaCapability,
    availability: Availability,
    lifecycle: catalog.Lifecycle,
    state: catalog.StateContract,
    specialization: Specialization,
    evidence: Evidence,
    evidence_note: []const u8,
};

const capabilities = [_]catalog.IsaCapability{
    .aarch64_asimd_fma,
    .aarch64_sve,
    .aarch64_sve2,
    .aarch64_sme,
    .aarch64_sme2,
    .x86_64_sse2,
    .x86_64_avx,
    .x86_64_fma,
    .x86_64_avx2,
    .x86_64_avx2_fma,
    .x86_64_avx512f,
    .x86_64_avx512f_fma,
};

const operations = [_]catalog.VectorOperation{
    .scal,
    .rscal,
    .copy,
    .swap,
    .axpy,
    .axpby,
    .dot,
    .dotu,
    .dotc,
    .dot_f32_acc_f64,
    .asum,
    .nrm2,
    .iamax,
    .rot,
    .rotm,
};

const scalars = [_]catalog.ScalarKind{ .f32, .f64, .complex_f32, .complex_f64 };

pub const entry_count = capabilities.len * 47;

pub fn entries() [entry_count]Entry {
    @setEvalBranchQuota(1_000_000);
    var result: [entry_count]Entry = undefined;
    var next: usize = 0;
    inline for (capabilities) |capability| {
        inline for (operations) |operation| {
            inline for (scalars) |scalar| {
                if (comptime applicable(operation, scalar)) {
                    result[next] = entryFor(operation, scalar, capability);
                    next += 1;
                }
            }
        }
    }
    std.debug.assert(next == result.len);
    return result;
}

fn applicable(operation: catalog.VectorOperation, scalar: catalog.ScalarKind) bool {
    return switch (operation) {
        .rscal, .dotu, .dotc => scalar == .complex_f32 or scalar == .complex_f64,
        .dot, .rotm => scalar == .f32 or scalar == .f64,
        .dot_f32_acc_f64 => scalar == .f32,
        .scal, .copy, .swap, .axpy, .axpby, .asum, .nrm2, .iamax, .rot => true,
        .rotg, .rotmg => false,
    };
}

fn entryFor(
    comptime operation: catalog.VectorOperation,
    comptime scalar: catalog.ScalarKind,
    comptime capability: catalog.IsaCapability,
) Entry {
    if (descriptorFor(operation, scalar, capability)) |descriptor| {
        return .{
            .stable_id = descriptor.name,
            .kernel = descriptor.kernel,
            .operation = operation,
            .scalar = scalar,
            .capability = capability,
            .availability = .implemented,
            .lifecycle = descriptor.lifecycle,
            .state = descriptor.state,
            .specialization = specializationFor(capability),
            .evidence = evidenceFor(descriptor.kernel),
            .evidence_note = noteFor(descriptor.kernel),
        };
    }
    const availability: Availability = if (capability == .x86_64_avx512f) .unsupported else .missing;
    return .{
        .stable_id = std.fmt.comptimePrint("level1.{s}.{s}.{s}.{s}", .{
            @tagName(operation),
            @tagName(scalar),
            @tagName(availability),
            @tagName(capability),
        }),
        .kernel = null,
        .operation = operation,
        .scalar = scalar,
        .capability = capability,
        .availability = availability,
        .lifecycle = .experimental,
        .state = missingState(operation, capability),
        .specialization = specializationFor(capability),
        .evidence = .{},
        .evidence_note = if (availability == .unsupported)
            "Zig/LLVM removes AVX-512F when FMA is disabled, so this is not an independently executable target tier"
        else
            "no distinct executable descriptor is registered for this operation/type/tier cell",
    };
}

fn descriptorFor(
    operation: catalog.VectorOperation,
    scalar: catalog.ScalarKind,
    capability: catalog.IsaCapability,
) ?catalog.Descriptor {
    for (catalog.registry.slice()) |descriptor| {
        if (descriptor.kernel.operation == operation and
            descriptor.kernel.scalar == scalar and
            descriptor.kernel.capability == capability and
            (descriptor.kernel.implementation == .aarch64_asimd or
                descriptor.kernel.implementation == .aarch64_sve or
                descriptor.kernel.implementation == .aarch64_sme2_streaming or
                descriptor.kernel.implementation == .x86_64_fixed_simd))
        {
            return descriptor;
        }
    }
    return null;
}

fn specializationFor(capability: catalog.IsaCapability) Specialization {
    return switch (capability) {
        .aarch64_asimd_fma,
        .aarch64_sve,
        .aarch64_sve2,
        .aarch64_sme,
        .aarch64_sme2,
        => .architecture_entrypoint,
        .x86_64_sse2,
        .x86_64_avx,
        .x86_64_fma,
        .x86_64_avx2,
        .x86_64_avx2_fma,
        .x86_64_avx512f,
        .x86_64_avx512f_fma,
        => .build_tier,
        else => unreachable,
    };
}

fn evidenceFor(kernel: catalog.KernelId) Evidence {
    return switch (kernel.implementation) {
        .aarch64_asimd => .{
            .build = true,
            .native_correctness = true,
            .native_performance = true,
        },
        .aarch64_sve => .{ .build = true },
        .aarch64_sme2_streaming => .{
            .build = true,
            .native_correctness = true,
            .native_performance = true,
        },
        .x86_64_fixed_simd => .{
            .build = true,
            .native_correctness = true,
            .native_performance = true,
        },
        else => .{},
    };
}

fn noteFor(kernel: catalog.KernelId) []const u8 {
    if (kernel.implementation == .aarch64_asimd and kernel.operation == .iamax and
        (kernel.scalar == .complex_f32 or kernel.scalar == .complex_f64))
    {
        return "forced-path correctness and fresh-process n=256/n=512 crossover evidence passed; production gate n<=256";
    }
    if (kernel.implementation == .x86_64_fixed_simd and
        kernel.capability == .x86_64_avx512f_fma and
        kernel.operation == .dot_f32_acc_f64)
    {
        return "AVX-512F/FMA forced-path correctness and same-ISA fresh-process evidence passed; production gate n>=65536";
    }
    if (kernel.implementation == .x86_64_fixed_simd and
        kernel.capability == .x86_64_avx512f_fma and
        kernel.operation == .iamax and
        (kernel.scalar == .complex_f32 or kernel.scalar == .complex_f64))
    {
        return "AVX-512F/FMA forced-path correctness and same-ISA fresh-process evidence passed; production gate n>=4096";
    }
    return switch (kernel.implementation) {
        .aarch64_asimd => "ASIMD-only fixed-candidate artifact contains no SME instructions; forced-path correctness and three-process native performance cover exact-8-KiB COPY and all operation families",
        .aarch64_sve => if (kernel.capability == .aarch64_sve2)
            "SVE2 cross-build reuses the executable base-SVE body; emulated forced-path correctness passed 29/29 at VQ=1/2/4/8, while native correctness and performance remain pending"
        else
            "SVE cross-build plus emulated forced-path correctness passed 29/29 at VQ=1/2/4/8; native correctness and performance remain pending",
        .aarch64_sme2_streaming => "SME2 streaming execution at 64-byte SVL passed balanced-state correctness; three-process arithmetic/swap and exact-8-KiB COPY measurements provide native performance evidence",
        .x86_64_fixed_simd => "separate x86 ISA-tier artifacts passed forced registry correctness and a four-size fresh-process all-operation sweep",
        else => "not an architecture coverage implementation",
    };
}

test "Level 1 architecture matrix exposes every applicable operation type and tier cell" {
    const report = entries();
    try std.testing.expectEqual(@as(usize, 12 * 47), report.len);
    for (report, 0..) |entry, index| {
        try std.testing.expect(entry.stable_id.len != 0);
        if (entry.availability != .implemented) try std.testing.expect(entry.kernel == null);
        if (entry.evidence.native_performance) try std.testing.expect(entry.evidence.native_correctness);
        if (entry.evidence.native_correctness) try std.testing.expect(entry.evidence.build);
        for (report[index + 1 ..]) |other| {
            try std.testing.expect(!std.mem.eql(u8, entry.stable_id, other.stable_id));
        }
    }
}

test "Level 1 matrix models SME and SME2 stateful cells independently" {
    const report = entries();
    const sme_copy = find(report, .copy, .complex_f64, .aarch64_sme).?;
    try std.testing.expectEqual(Availability.missing, sme_copy.availability);
    try std.testing.expectEqual(contract.StateKind.aarch64_streaming_sm, sme_copy.state.kind);

    const sme2_copy = find(report, .copy, .complex_f64, .aarch64_sme2).?;
    try std.testing.expectEqual(Availability.implemented, sme2_copy.availability);
    try std.testing.expectEqual(contract.StateKind.aarch64_streaming_sm, sme2_copy.state.kind);
    try std.testing.expect(sme2_copy.state.cleanup_on_success);
    try std.testing.expect(sme2_copy.state.cleanup_on_failure);

    const sme2_dot = find(report, .dot, .f64, .aarch64_sme2).?;
    try std.testing.expectEqual(Availability.implemented, sme2_dot.availability);
    try std.testing.expectEqual(contract.StateKind.aarch64_streaming_za, sme2_dot.state.kind);

    const sme2_nrm2 = find(report, .nrm2, .f64, .aarch64_sme2).?;
    try std.testing.expectEqual(Availability.missing, sme2_nrm2.availability);
    try std.testing.expectEqual(contract.StateKind.aarch64_streaming_za, sme2_nrm2.state.kind);
}

fn missingState(operation: catalog.VectorOperation, capability: catalog.IsaCapability) catalog.StateContract {
    if (capability != .aarch64_sme and capability != .aarch64_sme2) return .{};
    return contract.stateContract(switch (operation) {
        .copy, .swap => .aarch64_streaming_sm,
        else => .aarch64_streaming_za,
    });
}

test "Level 1 matrix distinguishes SVE2 inheritance and no-FMA x86 build tiers" {
    const report = entries();
    const sve = find(report, .dot, .f64, .aarch64_sve).?;
    try std.testing.expectEqual(Availability.implemented, sve.availability);
    try std.testing.expectEqual(catalog.Lifecycle.experimental, sve.lifecycle);
    try std.testing.expect(sve.evidence.build);
    try std.testing.expect(!sve.evidence.native_correctness);
    try std.testing.expect(!sve.evidence.native_performance);

    const sve2_dot = find(report, .dot, .f64, .aarch64_sve2).?;
    try std.testing.expectEqual(Availability.implemented, sve2_dot.availability);
    try std.testing.expect(sve2_dot.evidence.build);
    try std.testing.expect(!sve2_dot.evidence.native_correctness);

    const avx2 = find(report, .axpy, .f32, .x86_64_avx2).?;
    const avx2_fma = find(report, .axpy, .f32, .x86_64_avx2_fma).?;
    try std.testing.expectEqual(Availability.implemented, avx2.availability);
    try std.testing.expectEqual(Availability.implemented, avx2_fma.availability);
    try std.testing.expect(avx2.evidence.native_correctness);
    try std.testing.expect(avx2.evidence.native_performance);
    try std.testing.expect(!std.mem.eql(u8, avx2.stable_id, avx2_fma.stable_id));

    const avx512f = find(report, .axpy, .f32, .x86_64_avx512f).?;
    const avx512f_fma = find(report, .axpy, .f32, .x86_64_avx512f_fma).?;
    try std.testing.expectEqual(Availability.unsupported, avx512f.availability);
    try std.testing.expect(!avx512f.evidence.build);
    try std.testing.expect(avx512f.kernel == null);
    try std.testing.expectEqual(Availability.implemented, avx512f_fma.availability);

    const v3_mixed = find(report, .dot_f32_acc_f64, .f32, .x86_64_avx2_fma).?;
    const v4_mixed = find(report, .dot_f32_acc_f64, .f32, .x86_64_avx512f_fma).?;
    const v3_iamax = find(report, .iamax, .complex_f64, .x86_64_avx2_fma).?;
    const v4_iamax = find(report, .iamax, .complex_f64, .x86_64_avx512f_fma).?;
    try std.testing.expectEqual(catalog.Lifecycle.experimental, v3_mixed.lifecycle);
    try std.testing.expectEqual(catalog.Lifecycle.production, v4_mixed.lifecycle);
    try std.testing.expectEqual(catalog.Lifecycle.experimental, v3_iamax.lifecycle);
    try std.testing.expectEqual(catalog.Lifecycle.production, v4_iamax.lifecycle);
}

fn find(
    report: [entry_count]Entry,
    operation: catalog.VectorOperation,
    scalar: catalog.ScalarKind,
    capability: catalog.IsaCapability,
) ?Entry {
    for (report) |entry| {
        if (entry.operation == operation and entry.scalar == scalar and entry.capability == capability) return entry;
    }
    return null;
}
