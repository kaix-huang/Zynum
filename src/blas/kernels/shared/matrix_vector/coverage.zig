// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Test-enumerable Level 2 fused-body ISA coverage matrix.
//!
//! Each row is an operation/type/body/ISA cell. x86 rows describe separate
//! target-tier builds of the shared fixed-SIMD source rather than runtime
//! dispatch inside one fat binary. Missing rows remain explicit until a
//! distinct executable descriptor and forced-path proof exist.

const std = @import("std");

const catalog = @import("catalog.zig");

pub const Availability = enum {
    implemented,
    missing,
};

pub const Specialization = enum {
    architecture_entrypoint,
    build_tier,
    isolated_object,
};

pub const Evidence = packed struct {
    build: bool = false,
    native_correctness: bool = false,
    native_performance: bool = false,
};

pub const Entry = struct {
    stable_id: []const u8,
    kernel: ?catalog.KernelId,
    operation: catalog.Level2Operation,
    scalar: catalog.ScalarKind,
    implementation: catalog.Implementation,
    capability: catalog.IsaCapability,
    availability: Availability,
    lifecycle: catalog.Lifecycle,
    specialization: Specialization,
    evidence: Evidence,
    evidence_note: []const u8,
};

const capabilities = [_]catalog.IsaCapability{
    .aarch64_asimd_fma,
    .x86_64_sse2,
    .x86_64_avx,
    .x86_64_fma,
    .x86_64_avx2,
    .x86_64_avx2_fma,
    .x86_64_avx512f,
    .x86_64_avx512f_fma,
};

const implementations = [_]catalog.Implementation{
    .fused_gemv_no_trans,
    .fused_gemv_trans,
    .fused_gemv_conj_trans,
    .fused_rank_update,
    .fused_symmetric,
    .triangular_axpy,
    .triangular_dot,
};

const isolated_width_implementations = [_]catalog.Implementation{
    .fused_gemv_no_trans_narrow,
    .fused_gemv_trans_narrow,
    .fused_gemv_conj_trans_narrow,
    .fused_rank_update_narrow,
};

const operations = std.enums.values(catalog.Level2Operation);
const scalars = std.enums.values(catalog.ScalarKind);

pub const cells_per_capability = 36;
pub const isolated_width_entry_count = 13;
pub const entry_count = capabilities.len * cells_per_capability + isolated_width_entry_count;

pub fn entries() [entry_count]Entry {
    @setEvalBranchQuota(2_000_000);
    var result: [entry_count]Entry = undefined;
    var next: usize = 0;
    inline for (capabilities) |capability| {
        inline for (implementations) |implementation| {
            inline for (operations) |operation| {
                inline for (scalars) |scalar| {
                    if (comptime applicable(operation, scalar, implementation)) {
                        result[next] = entryFor(operation, scalar, implementation, capability);
                        next += 1;
                    }
                }
            }
        }
    }
    inline for (isolated_width_implementations) |implementation| {
        inline for (operations) |operation| {
            inline for (scalars) |scalar| {
                if (comptime applicable(operation, scalar, implementation)) {
                    result[next] = entryFor(operation, scalar, implementation, .x86_64_avx512f_fma);
                    next += 1;
                }
            }
        }
    }
    std.debug.assert(next == result.len);
    return result;
}

fn applicable(
    operation: catalog.Level2Operation,
    scalar: catalog.ScalarKind,
    implementation: catalog.Implementation,
) bool {
    return switch (implementation) {
        .fused_gemv_no_trans, .fused_gemv_trans => operation == .gemv,
        .fused_gemv_conj_trans => operation == .gemv and isComplex(scalar),
        .fused_rank_update => switch (operation) {
            .ger => isReal(scalar),
            .geru, .gerc => isComplex(scalar),
            else => false,
        },
        .fused_symmetric => (operation == .symv and isReal(scalar)) or
            (operation == .hemv and isComplex(scalar)),
        .triangular_axpy, .triangular_dot => (operation == .trmv or operation == .trsv),
        .fused_gemv_no_trans_narrow, .fused_gemv_trans_narrow => operation == .gemv and scalar != .f64,
        .fused_gemv_conj_trans_narrow => operation == .gemv and isComplex(scalar),
        .fused_rank_update_narrow => (operation == .ger and scalar == .f32) or
            ((operation == .geru or operation == .gerc) and isComplex(scalar)),
        else => false,
    };
}

fn isReal(scalar: catalog.ScalarKind) bool {
    return scalar == .f32 or scalar == .f64;
}

fn isComplex(scalar: catalog.ScalarKind) bool {
    return scalar == .complex_f32 or scalar == .complex_f64;
}

fn entryFor(
    comptime operation: catalog.Level2Operation,
    comptime scalar: catalog.ScalarKind,
    comptime implementation: catalog.Implementation,
    comptime capability: catalog.IsaCapability,
) Entry {
    if (catalog.findCapability(operation, scalar, implementation, capability)) |descriptor| {
        return .{
            .stable_id = descriptor.name,
            .kernel = descriptor.kernel,
            .operation = operation,
            .scalar = scalar,
            .implementation = implementation,
            .capability = capability,
            .availability = .implemented,
            .lifecycle = descriptor.lifecycle,
            .specialization = specializationFor(implementation, capability),
            .evidence = evidenceFor(operation, scalar, implementation, capability),
            .evidence_note = evidenceNote(operation, scalar, implementation, capability),
        };
    }
    return .{
        .stable_id = std.fmt.comptimePrint("level2.{s}.{s}.{s}.missing.{s}", .{
            @tagName(operation),
            @tagName(scalar),
            @tagName(implementation),
            @tagName(capability),
        }),
        .kernel = null,
        .operation = operation,
        .scalar = scalar,
        .implementation = implementation,
        .capability = capability,
        .availability = .missing,
        .lifecycle = .experimental,
        .specialization = specializationFor(implementation, capability),
        .evidence = .{},
        .evidence_note = "no distinct executable descriptor is registered for this Level 2 body/type/tier cell",
    };
}

fn evidenceFor(
    operation: catalog.Level2Operation,
    scalar: catalog.ScalarKind,
    implementation: catalog.Implementation,
    capability: catalog.IsaCapability,
) Evidence {
    return .{
        .build = true,
        .native_correctness = true,
        .native_performance = performanceTested(operation, scalar, implementation, capability),
    };
}

fn performanceTested(
    operation: catalog.Level2Operation,
    scalar: catalog.ScalarKind,
    implementation: catalog.Implementation,
    capability: catalog.IsaCapability,
) bool {
    if (isIsolatedWidth(implementation)) return true;
    if (capability == .aarch64_asimd_fma) return true;
    if (implementation == .fused_gemv_no_trans or
        implementation == .fused_gemv_trans or
        implementation == .fused_gemv_conj_trans)
    {
        return true;
    }
    if (implementation == .fused_rank_update and operation == .ger and isReal(scalar)) return true;
    return capability == .x86_64_avx512f_fma;
}

fn evidenceNote(
    operation: catalog.Level2Operation,
    scalar: catalog.ScalarKind,
    implementation: catalog.Implementation,
    capability: catalog.IsaCapability,
) []const u8 {
    if (isIsolatedWidth(implementation)) {
        return "same-layout isolated-object correctness and fresh-process candidate/control performance evidence passed; promotion remains shape/type selective";
    }
    if (capability == .aarch64_asimd_fma) {
        return "ASIMD/FMA forced-path correctness plus a 516-case, three-process dense, packed, banded, rank-update, symmetric, and triangular native performance sweep";
    }
    if (performanceTested(operation, scalar, implementation, capability)) {
        if (implementation == .fused_symmetric or implementation == .triangular_axpy or
            implementation == .triangular_dot or
            (implementation == .fused_rank_update and operation != .ger))
        {
            return "forced-path correctness and AVX-512F/FMA candidate controls passed; broad production promotion was rejected or narrowed where cases regressed";
        }
        return "forced-path correctness and fresh-process ISA-tier performance controls passed";
    }
    return "forced-path correctness passed for this ISA tier; a dedicated performance control is pending";
}

fn isIsolatedWidth(implementation: catalog.Implementation) bool {
    return switch (implementation) {
        .fused_gemv_no_trans_narrow,
        .fused_gemv_trans_narrow,
        .fused_gemv_conj_trans_narrow,
        .fused_rank_update_narrow,
        => true,
        else => false,
    };
}

fn specializationFor(implementation: catalog.Implementation, capability: catalog.IsaCapability) Specialization {
    if (isIsolatedWidth(implementation)) return .isolated_object;
    return if (capability == .aarch64_asimd_fma) .architecture_entrypoint else .build_tier;
}

test "Level 2 coverage matrix exposes implemented and missing fused ISA cells" {
    const report = entries();
    try std.testing.expectEqual(@as(usize, 8 * 36 + 13), report.len);
    var implemented: usize = 0;
    var missing: usize = 0;
    for (report, 0..) |entry, index| {
        try std.testing.expect(entry.stable_id.len != 0);
        if (entry.availability == .implemented) {
            implemented += 1;
            try std.testing.expect(entry.kernel != null);
            try std.testing.expect(entry.evidence.build);
        } else {
            missing += 1;
            try std.testing.expect(entry.kernel == null);
        }
        if (entry.evidence.native_performance) try std.testing.expect(entry.evidence.native_correctness);
        if (entry.evidence.native_correctness) try std.testing.expect(entry.evidence.build);
        for (report[index + 1 ..]) |other| {
            try std.testing.expect(!std.mem.eql(u8, entry.stable_id, other.stable_id));
        }
    }
    try std.testing.expectEqual(@as(usize, 8 * 36 + 13), implemented);
    try std.testing.expectEqual(@as(usize, 0), missing);
}

test "Level 2 coverage distinguishes x86 FMA tiers and structured gaps" {
    const report = entries();
    const avx2 = find(report, .gemv, .f32, .fused_gemv_no_trans, .x86_64_avx2).?;
    const avx2_fma = find(report, .gemv, .f32, .fused_gemv_no_trans, .x86_64_avx2_fma).?;
    try std.testing.expectEqual(Availability.implemented, avx2.availability);
    try std.testing.expectEqual(Availability.implemented, avx2_fma.availability);
    try std.testing.expect(!std.mem.eql(u8, avx2.stable_id, avx2_fma.stable_id));

    const complex_ger = find(report, .geru, .complex_f64, .fused_rank_update, .x86_64_avx512f_fma).?;
    const symv = find(report, .symv, .f64, .fused_symmetric, .aarch64_asimd_fma).?;
    const triangular = find(report, .trsv, .complex_f32, .triangular_dot, .x86_64_avx512f_fma).?;
    try std.testing.expectEqual(Availability.implemented, complex_ger.availability);
    try std.testing.expectEqual(catalog.Lifecycle.experimental, complex_ger.lifecycle);
    try std.testing.expectEqual(Availability.implemented, symv.availability);
    try std.testing.expectEqual(catalog.Lifecycle.experimental, symv.lifecycle);
    try std.testing.expectEqual(Availability.implemented, triangular.availability);
    try std.testing.expectEqual(catalog.Lifecycle.experimental, triangular.lifecycle);

    const narrow = find(report, .gemv, .f32, .fused_gemv_no_trans_narrow, .x86_64_avx512f_fma).?;
    try std.testing.expectEqual(Specialization.isolated_object, narrow.specialization);
    try std.testing.expect(narrow.evidence.native_performance);
}

fn find(
    report: [entry_count]Entry,
    operation: catalog.Level2Operation,
    scalar: catalog.ScalarKind,
    implementation: catalog.Implementation,
    capability: catalog.IsaCapability,
) ?Entry {
    for (report) |entry| {
        if (entry.operation == operation and
            entry.scalar == scalar and
            entry.implementation == implementation and
            entry.capability == capability)
        {
            return entry;
        }
    }
    return null;
}
