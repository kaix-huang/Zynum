// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Normalized, test-enumerable coverage across every BLAS kernel level.
//!
//! Level-specific coverage modules remain the owners of detailed contracts.
//! This module provides the small common projection used by generated reports
//! and completion gates. Evidence fields are deliberately independent:
//! building a cell does not imply native execution, and native correctness
//! does not imply a retained performance result.

const std = @import("std");

const contract = @import("contract.zig");
const vector_catalog = @import("shared/vector/catalog.zig");
const vector_coverage = @import("shared/vector/coverage.zig");
const matrix_vector_catalog = @import("shared/matrix_vector/catalog.zig");
const matrix_vector_coverage = @import("shared/matrix_vector/coverage.zig");
const gemm_catalog = @import("shared/matrix_matrix/catalog.zig");
const gemm_coverage = @import("shared/matrix_matrix/coverage.zig");
const real_gemm_executor = @import("shared/matrix_matrix/executor.zig");
const structured = @import("shared/matrix_matrix/structured_catalog.zig");
const complex_gemm_executor = @import("../core/matrix_matrix/gemm.zig");

pub const Level = enum {
    level1,
    level2,
    level3_gemm,
    level3_structured,
};

pub const Availability = enum {
    implemented,
    missing,
    rejected,
    unsupported,
};

pub const Evidence = packed struct {
    build: bool = false,
    native_correctness: bool = false,
    native_performance: bool = false,
};

pub const Entry = struct {
    level: Level,
    stable_id: []const u8,
    operation: []const u8,
    scalar: []const u8,
    implementation: []const u8,
    specialization: []const u8,
    capability: contract.IsaCapability,
    availability: Availability,
    lifecycle: contract.Lifecycle,
    state: contract.StateKind,
    evidence: Evidence,
    evidence_note: []const u8,
};

pub const Summary = struct {
    total: usize = 0,
    implemented: usize = 0,
    experimental: usize = 0,
    rejected: usize = 0,
    missing: usize = 0,
    unsupported: usize = 0,
    build_tested: usize = 0,
    native_correctness_tested: usize = 0,
    native_performance_tested: usize = 0,
};

pub const RegistryFamily = enum {
    level1,
    level2,
    real_gemm,
    complex_gemm,
    structured_level3,
};

/// Describes how a stable registry ID relates to execution.
/// Unbound states do not claim that the implementation body is absent; they
/// mean the production executor does not consume that stable ID as authority.
pub const ExecutorBinding = enum {
    bound_default,
    unbound_default_eligible,
    unbound_experimental,
    rejected_record,
};

/// Stable, normalized projection of every executable registry descriptor.
/// Coverage gaps remain in `Entry`; this projection records only registered
/// implementations and the relationships that must stay total.
pub const RegistryEntry = struct {
    family: RegistryFamily,
    stable_id: []const u8,
    lifecycle: contract.Lifecycle,
    fallback_id: ?[]const u8,
    executor_binding: ExecutorBinding,
};

pub const RegistrySummary = struct {
    total: usize = 0,
    portable_fallback: usize = 0,
    production: usize = 0,
    experimental: usize = 0,
    rejected: usize = 0,
    fallback_edges: usize = 0,
    maximum_fallback_depth: usize = 0,
    unique_ids: bool = true,
    legal_lifecycles: bool = true,
    fallback_chains_terminate: bool = true,
    bound_default: usize = 0,
    unbound_default_eligible: usize = 0,
    unbound_experimental: usize = 0,
    rejected_records: usize = 0,
    executor_binding_legal: bool = true,
    overall_executor_complete: bool = true,
    real_gemm_executor_complete: bool = true,
    complex_gemm_executor_complete: bool = true,
};

pub const registry_entry_count = vector_catalog.registered_descriptor_count +
    matrix_vector_catalog.descriptor_count +
    gemm_catalog.registered_descriptor_count +
    gemm_catalog.complex_registry.len +
    structured.registry.len;

pub fn registryEntries(streaming_vector_bytes: usize) [registry_entry_count]RegistryEntry {
    var result: [registry_entry_count]RegistryEntry = undefined;
    var next: usize = 0;

    for (vector_catalog.registry.slice()) |descriptor| {
        result[next] = .{
            .family = .level1,
            .stable_id = descriptor.name,
            .lifecycle = descriptor.lifecycle,
            .fallback_id = if (descriptor.fallback) |fallback| vectorName(fallback) else null,
            .executor_binding = bindingFor(descriptor.lifecycle, false),
        };
        next += 1;
    }
    for (matrix_vector_catalog.registry) |descriptor| {
        result[next] = .{
            .family = .level2,
            .stable_id = descriptor.name,
            .lifecycle = descriptor.lifecycle,
            .fallback_id = if (descriptor.fallback) |fallback| matrixVectorName(fallback) else null,
            .executor_binding = bindingFor(descriptor.lifecycle, false),
        };
        next += 1;
    }
    for (gemm_catalog.registeredDescriptors(streaming_vector_bytes)) |descriptor| {
        result[next] = .{
            .family = .real_gemm,
            .stable_id = descriptor.name,
            .lifecycle = descriptor.lifecycle,
            .fallback_id = if (descriptor.fallback == descriptor.kernel) null else @tagName(descriptor.fallback),
            .executor_binding = bindingFor(descriptor.lifecycle, real_gemm_executor.hasRegisteredExecutorMapping(descriptor.kernel)),
        };
        next += 1;
    }
    for (gemm_catalog.complex_registry) |descriptor| {
        result[next] = .{
            .family = .complex_gemm,
            .stable_id = descriptor.name,
            .lifecycle = descriptor.lifecycle,
            .fallback_id = if (descriptor.fallback) |fallback| complexGemmName(fallback) else null,
            .executor_binding = bindingFor(descriptor.lifecycle, complex_gemm_executor.hasComplexExecutorMapping(descriptor.kernel)),
        };
        next += 1;
    }
    for (structured.registry) |descriptor| {
        result[next] = .{
            .family = .structured_level3,
            .stable_id = descriptor.name,
            .lifecycle = descriptor.lifecycle,
            .fallback_id = if (descriptor.fallback) |fallback| @tagName(fallback) else null,
            .executor_binding = bindingFor(descriptor.lifecycle, false),
        };
        next += 1;
    }

    std.debug.assert(next == result.len);
    return result;
}

pub fn summarizeRegistry(registry: []const RegistryEntry) RegistrySummary {
    var summary: RegistrySummary = .{ .total = registry.len };
    for (registry, 0..) |entry, index| {
        switch (entry.lifecycle) {
            .portable_fallback => summary.portable_fallback += 1,
            .production => summary.production += 1,
            .experimental => summary.experimental += 1,
            .rejected => summary.rejected += 1,
        }
        summary.fallback_edges += @intFromBool(entry.fallback_id != null);
        switch (entry.executor_binding) {
            .bound_default => summary.bound_default += 1,
            .unbound_default_eligible => {
                summary.unbound_default_eligible += 1;
                summary.overall_executor_complete = false;
            },
            .unbound_experimental => {
                summary.unbound_experimental += 1;
                summary.overall_executor_complete = false;
            },
            .rejected_record => summary.rejected_records += 1,
        }
        summary.executor_binding_legal = summary.executor_binding_legal and executorBindingLegal(entry);
        if (entry.family == .real_gemm) {
            summary.real_gemm_executor_complete = summary.real_gemm_executor_complete and realGemmExecutorMapped(entry);
        }
        if (entry.family == .complex_gemm) {
            summary.complex_gemm_executor_complete = summary.complex_gemm_executor_complete and complexGemmExecutorMapped(entry);
        }
        summary.legal_lifecycles = summary.legal_lifecycles and lifecycleLegal(entry);

        for (registry[index + 1 ..]) |other| {
            if (entry.family == other.family and std.mem.eql(u8, entry.stable_id, other.stable_id)) {
                summary.unique_ids = false;
            }
        }

        var current = entry;
        var depth: usize = 0;
        while (current.fallback_id) |fallback_id| {
            depth += 1;
            if (depth > registry.len) {
                summary.fallback_chains_terminate = false;
                break;
            }
            current = findRegistryEntry(registry, current.family, fallback_id) orelse {
                summary.fallback_chains_terminate = false;
                break;
            };
        }
        summary.maximum_fallback_depth = @max(summary.maximum_fallback_depth, depth);
        if (current.fallback_id == null and current.lifecycle != .portable_fallback) {
            summary.fallback_chains_terminate = false;
        }
    }
    return summary;
}

fn findRegistryEntry(registry: []const RegistryEntry, family: RegistryFamily, stable_id: []const u8) ?RegistryEntry {
    for (registry) |entry| {
        if (entry.family == family and std.mem.eql(u8, entry.stable_id, stable_id)) return entry;
    }
    return null;
}

fn vectorName(kernel: vector_catalog.KernelId) []const u8 {
    for (vector_catalog.registry.slice()) |descriptor| {
        if (std.meta.eql(descriptor.kernel, kernel)) return descriptor.name;
    }
    unreachable;
}

fn matrixVectorName(kernel: matrix_vector_catalog.KernelId) []const u8 {
    for (matrix_vector_catalog.registry) |descriptor| {
        if (std.meta.eql(descriptor.kernel, kernel)) return descriptor.name;
    }
    unreachable;
}

fn complexGemmName(kernel: gemm_catalog.ComplexKernelId) []const u8 {
    for (gemm_catalog.complex_registry) |descriptor| {
        if (descriptor.kernel == kernel) return descriptor.name;
    }
    unreachable;
}

fn complexGemmKernel(stable_id: []const u8) ?gemm_catalog.ComplexKernelId {
    for (gemm_catalog.complex_registry) |descriptor| {
        if (std.mem.eql(u8, descriptor.name, stable_id)) return descriptor.kernel;
    }
    return null;
}

fn complexRegistryCoversEveryExecutorId(registry: []const gemm_catalog.ComplexDescriptor) bool {
    const kernel_ids = std.meta.tags(gemm_catalog.ComplexKernelId);
    if (registry.len != kernel_ids.len) return false;

    for (kernel_ids) |kernel| {
        var matches: usize = 0;
        for (registry) |descriptor| {
            matches += @intFromBool(descriptor.kernel == kernel);
        }
        if (matches != 1) return false;
    }
    return true;
}

comptime {
    if (!complexRegistryCoversEveryExecutorId(&gemm_catalog.complex_registry)) {
        @compileError("complex GEMM registry must contain every executor kernel id exactly once");
    }
}

fn lifecycleLegal(entry: RegistryEntry) bool {
    if (entry.fallback_id == null) return entry.lifecycle == .portable_fallback;
    return entry.lifecycle == .production or entry.lifecycle == .experimental or entry.lifecycle == .rejected;
}

fn bindingFor(lifecycle: contract.Lifecycle, runtime_id_bound: bool) ExecutorBinding {
    if (runtime_id_bound) return .bound_default;
    return switch (lifecycle) {
        .portable_fallback, .production => .unbound_default_eligible,
        .experimental => .unbound_experimental,
        .rejected => .rejected_record,
    };
}

fn executorBindingLegal(entry: RegistryEntry) bool {
    return switch (entry.executor_binding) {
        .bound_default => entry.lifecycle.defaultEligible(),
        .unbound_default_eligible => entry.lifecycle.defaultEligible(),
        .unbound_experimental => entry.lifecycle == .experimental,
        .rejected_record => entry.lifecycle == .rejected,
    };
}

fn realGemmExecutorMapped(entry: RegistryEntry) bool {
    if (entry.family != .real_gemm or entry.executor_binding != .bound_default) {
        return false;
    }
    const kernel = std.meta.stringToEnum(gemm_catalog.KernelId, entry.stable_id) orelse return false;
    return real_gemm_executor.hasRegisteredExecutorMapping(kernel);
}

fn complexGemmExecutorMapped(entry: RegistryEntry) bool {
    if (entry.family != .complex_gemm or entry.executor_binding != .bound_default) return false;
    const kernel = complexGemmKernel(entry.stable_id) orelse return false;
    return complex_gemm_executor.hasComplexExecutorMapping(kernel);
}

/// Freezes the current executor-binding state. Executor changes must update
/// this contract instead of silently turning catalog metadata into execution
/// evidence.
pub fn matchesExecutorBindingBaseline(summary: RegistrySummary) bool {
    return summary.bound_default == 26 and
        summary.unbound_default_eligible == 820 and
        summary.unbound_experimental == 368 and
        summary.rejected_records == 20 and
        summary.executor_binding_legal and
        !summary.overall_executor_complete and
        summary.real_gemm_executor_complete and
        summary.complex_gemm_executor_complete;
}

pub const entry_count = vector_coverage.entry_count +
    matrix_vector_coverage.entry_count +
    gemm_coverage.entry_count +
    structured.registry.len;

pub fn entries(streaming_vector_bytes: usize) [entry_count]Entry {
    var result: [entry_count]Entry = undefined;
    var next: usize = 0;

    const vector_entries = vector_coverage.entries();
    for (vector_entries) |entry| {
        result[next] = .{
            .level = .level1,
            .stable_id = entry.stable_id,
            .operation = @tagName(entry.operation),
            .scalar = @tagName(entry.scalar),
            .implementation = if (entry.kernel) |kernel| @tagName(kernel.implementation) else "unregistered",
            .specialization = @tagName(entry.specialization),
            .capability = entry.capability,
            .availability = switch (entry.availability) {
                .implemented => .implemented,
                .missing => .missing,
                .unsupported => .unsupported,
            },
            .lifecycle = entry.lifecycle,
            .state = entry.state.kind,
            .evidence = copyEvidence(entry.evidence),
            .evidence_note = entry.evidence_note,
        };
        next += 1;
    }

    const matrix_vector_entries = matrix_vector_coverage.entries();
    for (matrix_vector_entries) |entry| {
        result[next] = .{
            .level = .level2,
            .stable_id = entry.stable_id,
            .operation = @tagName(entry.operation),
            .scalar = @tagName(entry.scalar),
            .implementation = @tagName(entry.implementation),
            .specialization = @tagName(entry.specialization),
            .capability = entry.capability,
            .availability = if (entry.availability == .implemented) .implemented else .missing,
            .lifecycle = entry.lifecycle,
            .state = .none,
            .evidence = copyEvidence(entry.evidence),
            .evidence_note = entry.evidence_note,
        };
        next += 1;
    }

    const gemm_entries = gemm_coverage.entries(streaming_vector_bytes);
    for (gemm_entries) |entry| {
        result[next] = .{
            .level = .level3_gemm,
            .stable_id = entry.stable_id,
            .operation = "gemm",
            .scalar = @tagName(entry.scalar),
            .implementation = if (entry.kernel) |kernel| @tagName(kernel) else "unregistered",
            .specialization = "registry",
            .capability = entry.capability,
            .availability = switch (entry.availability) {
                .implemented => .implemented,
                .missing => .missing,
                .rejected => .rejected,
            },
            .lifecycle = entry.lifecycle,
            .state = entry.state.kind,
            .evidence = copyEvidence(entry.evidence),
            .evidence_note = entry.evidence_note,
        };
        next += 1;
    }

    for (structured.registry) |descriptor| {
        result[next] = .{
            .level = .level3_structured,
            .stable_id = descriptor.name,
            .operation = @tagName(descriptor.operation),
            .scalar = @tagName(descriptor.scalar),
            .implementation = @tagName(descriptor.implementation),
            .specialization = structuredSpecialization(descriptor.implementation),
            .capability = descriptor.capability,
            .availability = if (descriptor.lifecycle == .rejected) .rejected else .implemented,
            .lifecycle = descriptor.lifecycle,
            .state = descriptor.state.kind,
            .evidence = structuredEvidence(descriptor.implementation),
            .evidence_note = structuredEvidenceNote(descriptor.implementation),
        };
        next += 1;
    }

    std.debug.assert(next == result.len);
    return result;
}

fn copyEvidence(source: anytype) Evidence {
    return .{
        .build = source.build,
        .native_correctness = source.native_correctness,
        .native_performance = source.native_performance,
    };
}

fn structuredSpecialization(implementation: structured.Implementation) []const u8 {
    return switch (implementation) {
        .serial, .retained_column_parallel, .retained_left_column_parallel => "runtime_entrypoint",
        .rejected_dense_gemm, .rejected_right_row_parallel => "source_experiment",
        .isolated_dense_gemm, .isolated_right_row_parallel => "isolated_object",
        .blocked_rank_update,
        .blocked_symmetric_multiply,
        .blocked_triangular_left,
        .blocked_triangular_right,
        => "blocked_experimental",
    };
}

fn structuredEvidence(implementation: structured.Implementation) Evidence {
    return switch (implementation) {
        .blocked_rank_update,
        .blocked_symmetric_multiply,
        .blocked_triangular_left,
        .blocked_triangular_right,
        => .{ .build = true, .native_correctness = true },
        else => .{
            .build = true,
            .native_correctness = true,
            .native_performance = true,
        },
    };
}

fn structuredEvidenceNote(implementation: structured.Implementation) []const u8 {
    return switch (implementation) {
        .serial, .retained_column_parallel, .retained_left_column_parallel => "native correctness and fresh-process structured Level 3 sweeps cover the retained implementation",
        .rejected_dense_gemm, .rejected_right_row_parallel => "native correctness and regression evidence retained for the reverted in-graph experiment",
        .isolated_dense_gemm, .isolated_right_row_parallel => "same-layout isolated-object/control correctness and focused fresh-process performance evidence passed; lifecycle remains experimental",
        .blocked_rank_update,
        .blocked_symmetric_multiply,
        .blocked_triangular_left,
        .blocked_triangular_right,
        => "ReleaseSafe/ReleaseFast forced-path correctness passed; native performance promotion evidence remains pending",
    };
}

pub fn summarize(report: []const Entry) Summary {
    var summary: Summary = .{ .total = report.len };
    for (report) |entry| {
        switch (entry.availability) {
            .implemented => summary.implemented += 1,
            .rejected => summary.rejected += 1,
            .missing => summary.missing += 1,
            .unsupported => summary.unsupported += 1,
        }
        if (entry.availability == .implemented and entry.lifecycle == .experimental) {
            summary.experimental += 1;
        }
        summary.build_tested += @intFromBool(entry.evidence.build);
        summary.native_correctness_tested += @intFromBool(entry.evidence.native_correctness);
        summary.native_performance_tested += @intFromBool(entry.evidence.native_performance);
    }
    return summary;
}

test "consolidated coverage is stable unique and evidence-monotonic" {
    const report = entries(64);
    const summary = summarize(&report);
    var registry = registryEntries(64);
    const registry_summary = summarizeRegistry(&registry);
    try std.testing.expectEqual(entry_count, summary.total);
    try std.testing.expect(matchesExecutorBindingBaseline(registry_summary));
    try std.testing.expect(summary.implemented != 0);
    try std.testing.expect(summary.experimental != 0);
    try std.testing.expect(summary.rejected != 0);
    try std.testing.expect(summary.missing != 0);
    try std.testing.expect(summary.unsupported != 0);
    try std.testing.expect(summary.build_tested != 0);
    try std.testing.expect(summary.native_correctness_tested != 0);
    try std.testing.expect(summary.native_performance_tested != 0);
    try std.testing.expect(complexRegistryCoversEveryExecutorId(&gemm_catalog.complex_registry));
    try std.testing.expect(!complexRegistryCoversEveryExecutorId(
        gemm_catalog.complex_registry[0 .. gemm_catalog.complex_registry.len - 1],
    ));

    var duplicate_complex_registry = gemm_catalog.complex_registry;
    duplicate_complex_registry[duplicate_complex_registry.len - 1] = duplicate_complex_registry[0];
    try std.testing.expect(!complexRegistryCoversEveryExecutorId(&duplicate_complex_registry));
    try std.testing.expect(!real_gemm_executor.executorRouteMatchesKernel(
        .generic_f32_4x4,
        .generic_f64_4x4,
    ));
    try std.testing.expect(!complex_gemm_executor.complexExecutorRouteMatchesDescriptor(
        .compact_c32,
        .portable,
    ));

    for (report, 0..) |entry, index| {
        try std.testing.expect(entry.stable_id.len != 0);
        try std.testing.expect(entry.operation.len != 0);
        try std.testing.expect(entry.scalar.len != 0);
        if (entry.evidence.native_performance) try std.testing.expect(entry.evidence.native_correctness);
        if (entry.evidence.native_correctness) try std.testing.expect(entry.evidence.build);
        for (report[index + 1 ..]) |other| {
            if (entry.level == other.level) {
                try std.testing.expect(!std.mem.eql(u8, entry.stable_id, other.stable_id));
            }
        }
    }

    var saw_unbound_default = false;
    for (&registry) |*entry| {
        if (entry.executor_binding == .unbound_default_eligible) {
            saw_unbound_default = true;
            entry.executor_binding = .bound_default;
            break;
        }
    }
    try std.testing.expect(saw_unbound_default);
    try std.testing.expect(!matchesExecutorBindingBaseline(summarizeRegistry(&registry)));

    registry = registryEntries(64);
    var saw_complex = false;
    for (&registry) |*entry| {
        if (entry.family == .complex_gemm) {
            saw_complex = true;
            entry.executor_binding = .unbound_default_eligible;
            break;
        }
    }
    try std.testing.expect(saw_complex);
    try std.testing.expect(!matchesExecutorBindingBaseline(summarizeRegistry(&registry)));
}

test "consolidated coverage keeps complete SVE2 and remaining architecture gaps explicit" {
    const report = entries(64);
    var sve2_implemented: usize = 0;
    var sme2_implemented: usize = 0;
    var saw_sme_level1_gap = false;
    var saw_x86_amx_gap = false;
    var saw_isolated_structured = false;
    for (report) |entry| {
        if (entry.level == .level1 and entry.capability == .aarch64_sve2 and entry.availability == .implemented) {
            sve2_implemented += 1;
        }
        if (entry.level == .level1 and entry.capability == .aarch64_sme2 and entry.availability == .implemented) {
            sme2_implemented += 1;
            try std.testing.expect(entry.state == .aarch64_streaming_sm or entry.state == .aarch64_streaming_za);
        }
        if (entry.level == .level1 and entry.capability == .aarch64_sme and entry.availability == .missing) {
            saw_sme_level1_gap = true;
        }
        if (entry.level == .level3_gemm and entry.capability == .x86_64_amx and entry.availability == .missing) {
            saw_x86_amx_gap = true;
            try std.testing.expectEqual(contract.StateKind.x86_64_amx, entry.state);
        }
        if (entry.level == .level3_structured and std.mem.eql(u8, entry.specialization, "isolated_object")) {
            saw_isolated_structured = true;
            try std.testing.expectEqual(contract.Lifecycle.experimental, entry.lifecycle);
            try std.testing.expect(entry.evidence.native_performance);
        }
    }
    try std.testing.expectEqual(@as(usize, 47), sve2_implemented);
    try std.testing.expectEqual(@as(usize, 23), sme2_implemented);
    try std.testing.expect(saw_sme_level1_gap);
    try std.testing.expect(saw_x86_amx_gap);
    try std.testing.expect(saw_isolated_structured);
}
