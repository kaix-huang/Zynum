// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Static Level 2 semantic fallback catalog.
//!
//! This first registry layer names every public operation/type/storage cell and
//! its terminal scalar implementation. Architecture and fused unit-stride
//! descriptors are appended only when they map to distinct executable bodies;
//! facades and shape gates are not implementations.

const std = @import("std");
const contract = @import("../../contract.zig");

pub const OperationKind = contract.OperationKind;
pub const ScalarKind = contract.ScalarKind;
pub const IsaCapability = contract.IsaCapability;
pub const Lifecycle = contract.Lifecycle;
pub const TailStrategy = contract.TailStrategy;
pub const TailSupport = contract.TailSupport;
pub const VectorStrideContract = contract.VectorStrideContract;
pub const AliasContract = contract.AliasContract;
pub const LayoutSupport = contract.LayoutSupport;
pub const EpilogueSupport = contract.EpilogueSupport;
pub const StateContract = contract.StateContract;
pub const MatrixStorageKind = contract.MatrixStorageKind;
pub const StoredWindow = contract.StoredWindow;
pub const OutputOwnership = contract.OutputOwnership;
pub const FallbackAtomicity = contract.FallbackAtomicity;
pub const WorkspaceContract = contract.WorkspaceContract;
pub const TaskFallbackContract = contract.TaskFallbackContract;
pub const TraversalDependency = contract.TraversalDependency;

pub const Level2Operation = enum {
    gemv,
    gbmv,
    symv,
    hemv,
    sbmv,
    hbmv,
    spmv,
    hpmv,
    trmv,
    tbmv,
    tpmv,
    trsv,
    tbsv,
    tpsv,
    ger,
    geru,
    gerc,
    syr,
    her,
    spr,
    hpr,
    syr2,
    her2,
    spr2,
    hpr2,
};

pub const Implementation = enum {
    portable_scalar,
    core_unit,
    fused_gemv_no_trans,
    fused_gemv_trans,
    fused_gemv_conj_trans,
    fused_rank_update,
    fused_gemv_no_trans_narrow,
    fused_gemv_trans_narrow,
    fused_gemv_conj_trans_narrow,
    fused_rank_update_narrow,
    fused_symmetric,
    triangular_axpy,
    triangular_dot,
    compact_general_band,
    compact_symmetric_band,
    compact_symmetric_packed,
    compact_rank_packed,
    compact_triangular_band,
    compact_triangular_packed,
};

pub const BodyKind = enum {
    complete_operation,
    gemv_no_trans_panel,
    gemv_trans_panel,
    gemv_conj_trans_panel,
    rank_update_columns,
    symmetric_columns,
    triangular_axpy_step,
    triangular_dot_step,
    general_band_window,
    symmetric_band_window,
    packed_symmetric_columns,
    packed_rank_columns,
    triangular_band_operation,
    triangular_packed_operation,
};

pub const CompletionScope = enum {
    whole_operation,
    output_region,
    stored_columns,
    dependency_step,
};

pub const KernelId = struct {
    operation: Level2Operation,
    scalar: ScalarKind,
    implementation: Implementation,
    capability: IsaCapability,
};

pub const Descriptor = struct {
    name: []const u8,
    kernel: KernelId,
    fallback: ?KernelId,
    operation_kind: OperationKind,
    lifecycle: Lifecycle,
    tails: TailSupport,
    storage: MatrixStorageKind,
    stored_window: StoredWindow,
    strides: VectorStrideContract,
    aliasing: AliasContract,
    layouts: LayoutSupport,
    epilogue: EpilogueSupport,
    output: OutputOwnership,
    fallback_atomicity: FallbackAtomicity,
    workspace: WorkspaceContract,
    task_fallback: TaskFallbackContract,
    traversal: TraversalDependency,
    state: StateContract,
    body: BodyKind,
    completion: CompletionScope,
};

const operations = std.enums.values(Level2Operation);
const scalars = std.enums.values(ScalarKind);

const fixed_architecture_capabilities = [_]IsaCapability{
    .aarch64_asimd_fma,
    .x86_64_sse2,
    .x86_64_avx,
    .x86_64_fma,
    .x86_64_avx2,
    .x86_64_avx2_fma,
    .x86_64_avx512f,
    .x86_64_avx512f_fma,
};

const fixed_architecture_implementations = [_]Implementation{
    .fused_gemv_no_trans,
    .fused_gemv_trans,
    .fused_gemv_conj_trans,
    .fused_rank_update,
    .fused_symmetric,
    .triangular_axpy,
    .triangular_dot,
};

const isolated_width_implementations = [_]Implementation{
    .fused_gemv_no_trans_narrow,
    .fused_gemv_trans_narrow,
    .fused_gemv_conj_trans_narrow,
    .fused_rank_update_narrow,
};

pub const semantic_descriptor_count = 204;
pub const fixed_architecture_descriptor_count = fixed_architecture_capabilities.len * 36;
pub const isolated_width_descriptor_count = 13;
pub const descriptor_count = semantic_descriptor_count + fixed_architecture_descriptor_count + isolated_width_descriptor_count;
pub const registry = buildRegistry();

fn buildRegistry() [descriptor_count]Descriptor {
    @setEvalBranchQuota(2_000_000);
    var result: [descriptor_count]Descriptor = undefined;
    var next: usize = 0;
    inline for (std.enums.values(Implementation)) |implementation| {
        inline for (operations) |operation| {
            inline for (scalars) |scalar| {
                if (comptime implementationApplicable(operation, scalar, implementation)) {
                    result[next] = makeDescriptor(operation, scalar, implementation);
                    next += 1;
                }
            }
        }
    }
    inline for (fixed_architecture_capabilities) |capability| {
        inline for (fixed_architecture_implementations) |implementation| {
            inline for (operations) |operation| {
                inline for (scalars) |scalar| {
                    if (comptime fixedArchitectureApplicable(operation, scalar, implementation)) {
                        result[next] = makeArchitectureDescriptor(operation, scalar, implementation, capability);
                        next += 1;
                    }
                }
            }
        }
    }
    inline for (isolated_width_implementations) |implementation| {
        inline for (operations) |operation| {
            inline for (scalars) |scalar| {
                if (comptime isolatedWidthApplicable(operation, scalar, implementation)) {
                    result[next] = makeArchitectureDescriptor(operation, scalar, implementation, .x86_64_avx512f_fma);
                    next += 1;
                }
            }
        }
    }
    if (next != result.len) @compileError("Level 2 descriptor count is stale");
    validateRegistry(result);
    return result;
}

fn fixedArchitectureApplicable(
    operation: Level2Operation,
    scalar: ScalarKind,
    implementation: Implementation,
) bool {
    return switch (implementation) {
        .fused_gemv_no_trans, .fused_gemv_trans => operation == .gemv,
        .fused_gemv_conj_trans => operation == .gemv and
            (scalar == .complex_f32 or scalar == .complex_f64),
        .fused_rank_update => (operation == .ger and (scalar == .f32 or scalar == .f64)) or
            ((operation == .geru or operation == .gerc) and
                (scalar == .complex_f32 or scalar == .complex_f64)),
        .fused_symmetric => (operation == .symv and (scalar == .f32 or scalar == .f64)) or
            (operation == .hemv and (scalar == .complex_f32 or scalar == .complex_f64)),
        .triangular_axpy, .triangular_dot => operation == .trmv or operation == .trsv,
        .fused_gemv_no_trans_narrow,
        .fused_gemv_trans_narrow,
        .fused_gemv_conj_trans_narrow,
        .fused_rank_update_narrow,
        => isolatedWidthApplicable(operation, scalar, implementation),
        else => false,
    };
}

fn isolatedWidthApplicable(
    operation: Level2Operation,
    scalar: ScalarKind,
    implementation: Implementation,
) bool {
    return switch (implementation) {
        .fused_gemv_no_trans_narrow, .fused_gemv_trans_narrow => operation == .gemv and scalar != .f64,
        .fused_gemv_conj_trans_narrow => operation == .gemv and
            (scalar == .complex_f32 or scalar == .complex_f64),
        .fused_rank_update_narrow => (operation == .ger and scalar == .f32) or
            ((operation == .geru or operation == .gerc) and
                (scalar == .complex_f32 or scalar == .complex_f64)),
        else => false,
    };
}

fn implementationApplicable(
    operation: Level2Operation,
    scalar: ScalarKind,
    implementation: Implementation,
) bool {
    if (!applicable(operation, scalar)) return false;
    return switch (implementation) {
        .portable_scalar, .core_unit => true,
        .fused_gemv_no_trans, .fused_gemv_trans => operation == .gemv,
        .fused_gemv_conj_trans => operation == .gemv and
            (scalar == .complex_f32 or scalar == .complex_f64),
        .fused_rank_update => operation == .ger or operation == .geru or operation == .gerc,
        .fused_gemv_no_trans_narrow,
        .fused_gemv_trans_narrow,
        .fused_gemv_conj_trans_narrow,
        .fused_rank_update_narrow,
        => false,
        .fused_symmetric => operation == .symv or operation == .hemv,
        .triangular_axpy, .triangular_dot => operation == .trmv or operation == .trsv,
        .compact_general_band => operation == .gbmv,
        .compact_symmetric_band => operation == .sbmv or operation == .hbmv,
        .compact_symmetric_packed => operation == .spmv or operation == .hpmv,
        .compact_rank_packed => operation == .spr or operation == .hpr or
            operation == .spr2 or operation == .hpr2,
        .compact_triangular_band => operation == .tbmv or operation == .tbsv,
        .compact_triangular_packed => operation == .tpmv or operation == .tpsv,
    };
}

fn applicable(operation: Level2Operation, scalar: ScalarKind) bool {
    return switch (operation) {
        .gemv, .gbmv, .trmv, .tbmv, .tpmv, .trsv, .tbsv, .tpsv => true,
        .symv, .sbmv, .spmv, .ger, .syr, .spr, .syr2, .spr2 => scalar == .f32 or scalar == .f64,
        .hemv, .hbmv, .hpmv, .geru, .gerc, .her, .hpr, .her2, .hpr2 => scalar == .complex_f32 or scalar == .complex_f64,
    };
}

fn makeDescriptor(
    comptime operation: Level2Operation,
    comptime scalar: ScalarKind,
    comptime implementation: Implementation,
) Descriptor {
    return makeDescriptorWithCapability(operation, scalar, implementation, .generic);
}

fn makeArchitectureDescriptor(
    comptime operation: Level2Operation,
    comptime scalar: ScalarKind,
    comptime implementation: Implementation,
    comptime capability: IsaCapability,
) Descriptor {
    return makeDescriptorWithCapability(operation, scalar, implementation, capability);
}

fn makeDescriptorWithCapability(
    comptime operation: Level2Operation,
    comptime scalar: ScalarKind,
    comptime implementation: Implementation,
    comptime capability: IsaCapability,
) Descriptor {
    return .{
        .name = std.fmt.comptimePrint("level2.{s}.{s}.{s}.{s}", .{
            @tagName(operation),
            @tagName(scalar),
            @tagName(implementation),
            @tagName(capability),
        }),
        .kernel = .{
            .operation = operation,
            .scalar = scalar,
            .implementation = implementation,
            .capability = capability,
        },
        .fallback = fallbackFor(operation, scalar, implementation, capability),
        .operation_kind = operationKind(operation),
        .lifecycle = lifecycleFor(operation, scalar, implementation, capability),
        .tails = tailSupport(implementation, capability),
        .storage = storageFor(operation),
        .stored_window = storedWindowFor(operation),
        .strides = strideContract(operation, implementation),
        .aliasing = .blas_valid,
        .layouts = layoutSupport(operation, scalar, implementation),
        .epilogue = epilogueSupport(operation, implementation),
        .output = outputOwnership(operation, implementation),
        .fallback_atomicity = if (isSpecialized(implementation)) .reject_before_write else .completes_operation,
        .workspace = workspaceContract(operation, implementation),
        .task_fallback = taskFallbackContract(implementation),
        .traversal = traversalDependency(operation),
        .state = .{},
        .body = bodyKind(implementation),
        .completion = completionScope(operation, implementation),
    };
}

fn lifecycleFor(
    operation: Level2Operation,
    scalar: ScalarKind,
    implementation: Implementation,
    capability: IsaCapability,
) Lifecycle {
    _ = operation;
    if (implementation == .portable_scalar) return .portable_fallback;
    if (capability != .generic and
        (((implementation == .fused_rank_update or implementation == .fused_rank_update_narrow) and
            (scalar == .complex_f32 or scalar == .complex_f64)) or
            implementation == .fused_symmetric or
            implementation == .triangular_axpy or implementation == .triangular_dot))
    {
        return .experimental;
    }
    return .production;
}

fn tailSupport(implementation: Implementation, capability: IsaCapability) TailSupport {
    const fixed_architecture = capability != .generic;
    const vector_body = switch (implementation) {
        .fused_gemv_no_trans,
        .fused_gemv_trans,
        .fused_gemv_conj_trans,
        .fused_rank_update,
        .fused_gemv_no_trans_narrow,
        .fused_gemv_trans_narrow,
        .fused_gemv_conj_trans_narrow,
        .fused_rank_update_narrow,
        .fused_symmetric,
        .triangular_axpy,
        .triangular_dot,
        => true,
        else => false,
    };
    const dimension_tail: TailStrategy = if (fixed_architecture or vector_body) .scalar_cleanup else .native;
    return .{
        .m = dimension_tail,
        .n = dimension_tail,
        .k = .native,
    };
}

fn isSpecialized(implementation: Implementation) bool {
    return implementation != .portable_scalar and implementation != .core_unit;
}

fn isPartialBody(implementation: Implementation) bool {
    return switch (implementation) {
        .fused_gemv_no_trans,
        .fused_gemv_trans,
        .fused_gemv_conj_trans,
        .fused_rank_update,
        .fused_gemv_no_trans_narrow,
        .fused_gemv_trans_narrow,
        .fused_gemv_conj_trans_narrow,
        .fused_rank_update_narrow,
        .fused_symmetric,
        .triangular_axpy,
        .triangular_dot,
        .compact_symmetric_packed,
        .compact_rank_packed,
        => true,
        else => false,
    };
}

fn fallbackFor(
    operation: Level2Operation,
    scalar: ScalarKind,
    implementation: Implementation,
    capability: IsaCapability,
) ?KernelId {
    if (implementation == .portable_scalar) return null;
    if (capability != .generic) return .{
        .operation = operation,
        .scalar = scalar,
        .implementation = .core_unit,
        .capability = .generic,
    };
    return .{
        .operation = operation,
        .scalar = scalar,
        .implementation = if (implementation == .core_unit) .portable_scalar else .core_unit,
        .capability = .generic,
    };
}

fn bodyKind(implementation: Implementation) BodyKind {
    return switch (implementation) {
        .portable_scalar, .core_unit => .complete_operation,
        .fused_gemv_no_trans, .fused_gemv_no_trans_narrow => .gemv_no_trans_panel,
        .fused_gemv_trans, .fused_gemv_trans_narrow => .gemv_trans_panel,
        .fused_gemv_conj_trans, .fused_gemv_conj_trans_narrow => .gemv_conj_trans_panel,
        .fused_rank_update, .fused_rank_update_narrow => .rank_update_columns,
        .fused_symmetric => .symmetric_columns,
        .triangular_axpy => .triangular_axpy_step,
        .triangular_dot => .triangular_dot_step,
        .compact_general_band => .general_band_window,
        .compact_symmetric_band => .symmetric_band_window,
        .compact_symmetric_packed => .packed_symmetric_columns,
        .compact_rank_packed => .packed_rank_columns,
        .compact_triangular_band => .triangular_band_operation,
        .compact_triangular_packed => .triangular_packed_operation,
    };
}

fn completionScope(operation: Level2Operation, implementation: Implementation) CompletionScope {
    _ = operation;
    if (!isPartialBody(implementation)) return .whole_operation;
    return switch (implementation) {
        .fused_gemv_no_trans,
        .fused_gemv_trans,
        .fused_gemv_conj_trans,
        .fused_gemv_no_trans_narrow,
        .fused_gemv_trans_narrow,
        .fused_gemv_conj_trans_narrow,
        .fused_symmetric,
        .compact_symmetric_packed,
        => .output_region,
        .fused_rank_update, .fused_rank_update_narrow, .compact_rank_packed => .stored_columns,
        .triangular_axpy, .triangular_dot => .dependency_step,
        else => unreachable,
    };
}

fn operationKind(operation: Level2Operation) OperationKind {
    return switch (operation) {
        .gemv,
        .gbmv,
        .symv,
        .hemv,
        .sbmv,
        .hbmv,
        .spmv,
        .hpmv,
        .trmv,
        .tbmv,
        .tpmv,
        .trsv,
        .tbsv,
        .tpsv,
        => .matrix_vector,
        .ger,
        .geru,
        .gerc,
        .syr,
        .her,
        .spr,
        .hpr,
        .syr2,
        .her2,
        .spr2,
        .hpr2,
        => .matrix_rank_update,
    };
}

fn storageFor(operation: Level2Operation) MatrixStorageKind {
    return switch (operation) {
        .gemv, .ger, .geru, .gerc => .dense_general,
        .gbmv => .general_band,
        .symv, .hemv, .trmv, .trsv, .syr, .her, .syr2, .her2 => .dense_triangle,
        .sbmv, .hbmv => .symmetric_band,
        .tbmv, .tbsv => .triangular_band,
        .spmv, .hpmv, .tpmv, .tpsv, .spr, .hpr, .spr2, .hpr2 => .packed_triangle,
    };
}

fn storedWindowFor(operation: Level2Operation) StoredWindow {
    return switch (storageFor(operation)) {
        .dense_general => .full_matrix,
        .dense_triangle => .selected_triangle,
        .general_band, .symmetric_band, .triangular_band => .exact_band,
        .packed_triangle => .packed_columns,
        .not_applicable => unreachable,
    };
}

fn strideContract(operation: Level2Operation, implementation: Implementation) VectorStrideContract {
    const unary_rank = operation == .syr or operation == .her or operation == .spr or operation == .hpr;
    const triangular = isTriangular(operation);
    const stride_rule: contract.VectorStrideRule = if (implementation == .portable_scalar) .any_nonzero else .unit;
    return .{
        .x = stride_rule,
        .y = if (unary_rank or triangular) .not_applicable else stride_rule,
    };
}

fn layoutSupport(operation: Level2Operation, scalar: ScalarKind, implementation: Implementation) LayoutSupport {
    switch (implementation) {
        .fused_gemv_no_trans, .fused_gemv_no_trans_narrow, .triangular_axpy => return .{},
        .fused_gemv_trans, .fused_gemv_trans_narrow => return .{ .no_trans = false, .transposed_a = true },
        .fused_gemv_conj_trans, .fused_gemv_conj_trans_narrow => return .{
            .no_trans = false,
            .transposed_a = true,
            .conjugated_a = true,
        },
        .triangular_dot => return .{
            .no_trans = false,
            .transposed_a = true,
            .conjugated_a = scalar == .complex_f32 or scalar == .complex_f64,
        },
        else => {},
    }
    if (operation == .gemv or operation == .gbmv or isTriangular(operation)) {
        return .{
            .transposed_a = true,
            .conjugated_a = scalar == .complex_f32 or scalar == .complex_f64,
        };
    }
    return .{};
}

fn epilogueSupport(operation: Level2Operation, implementation: Implementation) EpilogueSupport {
    if (isTriangular(operation)) {
        return .{
            .arbitrary_alpha = false,
            .arbitrary_beta = false,
            .alpha_zero = false,
            .beta_zero = false,
        };
    }
    if (operationKind(operation) == .matrix_rank_update) {
        return .{
            .arbitrary_beta = false,
            .beta_zero = false,
        };
    }
    if (isPartialBody(implementation)) {
        return .{
            .arbitrary_beta = false,
            .beta_zero = false,
        };
    }
    return .{};
}

fn outputOwnership(operation: Level2Operation, implementation: Implementation) OutputOwnership {
    if (isTriangular(operation)) return .in_place_dependency_vector;
    switch (implementation) {
        .fused_gemv_no_trans,
        .fused_gemv_trans,
        .fused_gemv_conj_trans,
        .fused_gemv_no_trans_narrow,
        .fused_gemv_trans_narrow,
        .fused_gemv_conj_trans_narrow,
        => return .additive_vector_region,
        .fused_symmetric, .compact_symmetric_packed => return .private_vector_delta,
        .fused_rank_update, .fused_rank_update_narrow, .compact_rank_packed => return .stored_matrix_columns,
        else => {},
    }
    return switch (operationKind(operation)) {
        .matrix_vector => .final_vector,
        .matrix_rank_update => switch (storageFor(operation)) {
            .dense_general => .full_general_matrix,
            else => .stored_matrix_columns,
        },
        else => unreachable,
    };
}

fn traversalDependency(operation: Level2Operation) TraversalDependency {
    return if (isTriangular(operation)) .triangular_ordered else .none;
}

fn workspaceContract(operation: Level2Operation, implementation: Implementation) WorkspaceContract {
    if (implementation == .portable_scalar) return .{};
    if (implementation == .fused_symmetric or implementation == .compact_symmetric_packed) return .{
        .private_output = true,
        .merge_required = true,
    };
    if (isSpecialized(implementation)) return .{};
    return switch (operation) {
        .gemv, .symv, .hemv, .spmv, .hpmv => .{
            .private_output = true,
            .merge_required = true,
        },
        else => .{},
    };
}

fn taskFallbackContract(implementation: Implementation) TaskFallbackContract {
    return switch (implementation) {
        .compact_symmetric_packed => .private_results_commit_after_all_tasks,
        .compact_rank_packed => .disjoint_outputs_all_tasks_required,
        else => .not_applicable,
    };
}

fn isTriangular(operation: Level2Operation) bool {
    return switch (operation) {
        .trmv, .tbmv, .tpmv, .trsv, .tbsv, .tpsv => true,
        else => false,
    };
}

fn validateRegistry(items: [descriptor_count]Descriptor) void {
    for (items, 0..) |descriptor, index| {
        if (descriptor.name.len == 0) @compileError("Level 2 descriptor name is empty");
        if (descriptor.kernel.implementation == .portable_scalar) {
            if (descriptor.lifecycle != .portable_fallback or descriptor.fallback != null) {
                @compileError("terminal Level 2 descriptor has an invalid fallback contract");
            }
        } else {
            if ((descriptor.lifecycle != .production and descriptor.lifecycle != .experimental) or
                descriptor.fallback == null)
            {
                @compileError("unit Level 2 descriptor has an invalid fallback contract");
            }
            if (!containsKernel(items, descriptor.fallback.?)) @compileError("Level 2 fallback is absent");
        }
        if (descriptor.kernel.capability == .generic) {
            // Semantic execution layers and generic fused bodies remain
            // target-independent. Architecture leaves are appended below.
        } else if (!isFixedArchitectureCapability(descriptor.kernel.capability) or
            !fixedArchitectureApplicable(
                descriptor.kernel.operation,
                descriptor.kernel.scalar,
                descriptor.kernel.implementation,
            ))
        {
            @compileError("Level 2 architecture descriptor has an invalid capability or body");
        }
        if (descriptor.output == .in_place_dependency_vector and descriptor.traversal != .triangular_ordered) {
            @compileError("in-place Level 2 descriptor lacks its traversal dependency");
        }
        if (descriptor.output == .private_vector_delta and
            (!descriptor.workspace.private_output or !descriptor.workspace.merge_required))
        {
            @compileError("private Level 2 output lacks its merge contract");
        }
        if (descriptor.completion == .dependency_step and descriptor.traversal != .triangular_ordered) {
            @compileError("Level 2 dependency body lacks ordered traversal");
        }
        if (isSpecialized(descriptor.kernel.implementation) and
            descriptor.fallback_atomicity != .reject_before_write)
        {
            @compileError("specialized Level 2 body does not reject before fallback");
        }
        if (descriptor.task_fallback == .private_results_commit_after_all_tasks and
            (!descriptor.workspace.private_output or !descriptor.workspace.merge_required))
        {
            @compileError("private Level 2 task composition lacks an atomic merge buffer");
        }
        if (descriptor.task_fallback == .disjoint_outputs_all_tasks_required and
            descriptor.output != .stored_matrix_columns)
        {
            @compileError("disjoint Level 2 task composition lacks stored-column ownership");
        }
        for (items[index + 1 ..]) |other| {
            if (std.mem.eql(u8, descriptor.name, other.name)) @compileError("duplicate Level 2 descriptor name");
            if (std.meta.eql(descriptor.kernel, other.kernel)) @compileError("duplicate Level 2 kernel id");
        }
    }
}

fn find(operation: Level2Operation, scalar: ScalarKind) ?Descriptor {
    for (registry) |descriptor| {
        if (descriptor.kernel.operation == operation and descriptor.kernel.scalar == scalar) return descriptor;
    }
    return null;
}

pub fn findImplementation(operation: Level2Operation, scalar: ScalarKind, implementation: Implementation) ?Descriptor {
    for (registry) |descriptor| {
        if (descriptor.kernel.operation == operation and
            descriptor.kernel.scalar == scalar and
            descriptor.kernel.implementation == implementation)
        {
            return descriptor;
        }
    }
    return null;
}

pub fn findCapability(
    operation: Level2Operation,
    scalar: ScalarKind,
    implementation: Implementation,
    capability: IsaCapability,
) ?Descriptor {
    for (registry) |descriptor| {
        if (descriptor.kernel.operation == operation and
            descriptor.kernel.scalar == scalar and
            descriptor.kernel.implementation == implementation and
            descriptor.kernel.capability == capability)
        {
            return descriptor;
        }
    }
    return null;
}

fn isFixedArchitectureCapability(capability: IsaCapability) bool {
    inline for (fixed_architecture_capabilities) |candidate| {
        if (capability == candidate) return true;
    }
    return false;
}

fn containsKernel(items: [descriptor_count]Descriptor, kernel: KernelId) bool {
    for (items) |descriptor| {
        if (std.meta.eql(descriptor.kernel, kernel)) return true;
    }
    return false;
}

test "Level 2 terminal catalog covers every public operation and scalar cell" {
    try std.testing.expectEqual(@as(usize, 505), registry.len);
    inline for (operations) |operation| {
        inline for (scalars) |scalar| {
            try std.testing.expectEqual(applicable(operation, scalar), findImplementation(operation, scalar, .portable_scalar) != null);
            try std.testing.expectEqual(applicable(operation, scalar), findImplementation(operation, scalar, .core_unit) != null);
        }
    }
}

test "Level 2 fixed SIMD build tiers have distinct identities and scalar tails" {
    const asimd = findCapability(.gemv, .f32, .fused_gemv_no_trans, .aarch64_asimd_fma).?;
    const sse2 = findCapability(.gemv, .f32, .fused_gemv_no_trans, .x86_64_sse2).?;
    const avx2 = findCapability(.gemv, .f32, .fused_gemv_no_trans, .x86_64_avx2).?;
    const avx2_fma = findCapability(.gemv, .f32, .fused_gemv_no_trans, .x86_64_avx2_fma).?;
    const avx512 = findCapability(.gemv, .f32, .fused_gemv_no_trans, .x86_64_avx512f_fma).?;
    try std.testing.expect(!std.meta.eql(asimd.kernel, sse2.kernel));
    try std.testing.expect(!std.meta.eql(avx2.kernel, avx2_fma.kernel));
    try std.testing.expect(!std.meta.eql(avx2_fma.kernel, avx512.kernel));
    try std.testing.expectEqual(TailStrategy.scalar_cleanup, avx512.tails.m);
    try std.testing.expectEqual(TailStrategy.scalar_cleanup, avx512.tails.n);
    try std.testing.expectEqual(Implementation.core_unit, avx512.fallback.?.implementation);
    try std.testing.expectEqual(IsaCapability.generic, avx512.fallback.?.capability);

    try std.testing.expect(findCapability(.ger, .f64, .fused_rank_update, .x86_64_avx512f) != null);
    const complex_ger = findCapability(.geru, .complex_f64, .fused_rank_update, .x86_64_avx512f).?;
    try std.testing.expectEqual(Lifecycle.experimental, complex_ger.lifecycle);
    try std.testing.expect(findCapability(.gemv, .complex_f64, .fused_gemv_conj_trans, .aarch64_asimd_fma) != null);
}

test "isolated x86 width bodies have distinct identities and lifecycle" {
    const wide = findCapability(.gemv, .f32, .fused_gemv_no_trans, .x86_64_avx512f_fma).?;
    const narrow = findCapability(.gemv, .f32, .fused_gemv_no_trans_narrow, .x86_64_avx512f_fma).?;
    try std.testing.expect(!std.meta.eql(wide.kernel, narrow.kernel));
    try std.testing.expectEqual(Lifecycle.production, narrow.lifecycle);
    try std.testing.expectEqual(BodyKind.gemv_no_trans_panel, narrow.body);
    try std.testing.expectEqual(Implementation.core_unit, narrow.fallback.?.implementation);

    const complex_rank = findCapability(.gerc, .complex_f64, .fused_rank_update_narrow, .x86_64_avx512f_fma).?;
    try std.testing.expectEqual(Lifecycle.experimental, complex_rank.lifecycle);
}

test "Level 2 fused bodies declare partial ownership and unit fallback" {
    const gemv_c = findImplementation(.gemv, .complex_f64, .fused_gemv_conj_trans).?;
    try std.testing.expectEqual(BodyKind.gemv_conj_trans_panel, gemv_c.body);
    try std.testing.expect(!gemv_c.layouts.no_trans);
    try std.testing.expect(gemv_c.layouts.transposed_a);
    try std.testing.expect(gemv_c.layouts.conjugated_a);
    try std.testing.expectEqual(OutputOwnership.additive_vector_region, gemv_c.output);
    try std.testing.expectEqual(Implementation.core_unit, gemv_c.fallback.?.implementation);
    try std.testing.expectEqual(FallbackAtomicity.reject_before_write, gemv_c.fallback_atomicity);

    const hemv = findImplementation(.hemv, .complex_f32, .fused_symmetric).?;
    try std.testing.expectEqual(OutputOwnership.private_vector_delta, hemv.output);
    try std.testing.expect(hemv.workspace.private_output);
    try std.testing.expect(hemv.workspace.merge_required);

    const trsv_dot = findImplementation(.trsv, .complex_f64, .triangular_dot).?;
    try std.testing.expectEqual(CompletionScope.dependency_step, trsv_dot.completion);
    try std.testing.expectEqual(TraversalDependency.triangular_ordered, trsv_dot.traversal);
}

test "Level 2 storage and ownership contracts distinguish compact and dependency paths" {
    const gbmv = find(.gbmv, .complex_f64).?;
    try std.testing.expectEqual(MatrixStorageKind.general_band, gbmv.storage);
    try std.testing.expectEqual(StoredWindow.exact_band, gbmv.stored_window);
    try std.testing.expectEqual(OutputOwnership.final_vector, gbmv.output);

    const tpsv = find(.tpsv, .f64).?;
    try std.testing.expectEqual(MatrixStorageKind.packed_triangle, tpsv.storage);
    try std.testing.expectEqual(StoredWindow.packed_columns, tpsv.stored_window);
    try std.testing.expectEqual(OutputOwnership.in_place_dependency_vector, tpsv.output);
    try std.testing.expectEqual(TraversalDependency.triangular_ordered, tpsv.traversal);

    const her2 = find(.her2, .complex_f32).?;
    try std.testing.expectEqual(MatrixStorageKind.dense_triangle, her2.storage);
    try std.testing.expectEqual(OutputOwnership.stored_matrix_columns, her2.output);
}

test "Level 2 compact families declare window merge and task fallback contracts" {
    const gbmv = findImplementation(.gbmv, .complex_f64, .compact_general_band).?;
    try std.testing.expectEqual(BodyKind.general_band_window, gbmv.body);
    try std.testing.expectEqual(StoredWindow.exact_band, gbmv.stored_window);
    try std.testing.expectEqual(CompletionScope.whole_operation, gbmv.completion);

    const hpmv = findImplementation(.hpmv, .complex_f32, .compact_symmetric_packed).?;
    try std.testing.expectEqual(BodyKind.packed_symmetric_columns, hpmv.body);
    try std.testing.expectEqual(StoredWindow.packed_columns, hpmv.stored_window);
    try std.testing.expectEqual(OutputOwnership.private_vector_delta, hpmv.output);
    try std.testing.expect(hpmv.workspace.private_output);
    try std.testing.expect(hpmv.workspace.merge_required);
    try std.testing.expectEqual(
        TaskFallbackContract.private_results_commit_after_all_tasks,
        hpmv.task_fallback,
    );

    const hpr2 = findImplementation(.hpr2, .complex_f64, .compact_rank_packed).?;
    try std.testing.expectEqual(CompletionScope.stored_columns, hpr2.completion);
    try std.testing.expectEqual(OutputOwnership.stored_matrix_columns, hpr2.output);
    try std.testing.expectEqual(
        TaskFallbackContract.disjoint_outputs_all_tasks_required,
        hpr2.task_fallback,
    );

    const tbsv = findImplementation(.tbsv, .f64, .compact_triangular_band).?;
    try std.testing.expectEqual(StoredWindow.exact_band, tbsv.stored_window);
    try std.testing.expectEqual(OutputOwnership.in_place_dependency_vector, tbsv.output);
    try std.testing.expectEqual(TraversalDependency.triangular_ordered, tbsv.traversal);

    const tpmv = findImplementation(.tpmv, .complex_f32, .compact_triangular_packed).?;
    try std.testing.expectEqual(StoredWindow.packed_columns, tpmv.stored_window);
    try std.testing.expectEqual(CompletionScope.whole_operation, tpmv.completion);
    try std.testing.expectEqual(Implementation.core_unit, tpmv.fallback.?.implementation);
}

test "Level 2 unit execution layers require unit stride and total fallback" {
    const unit = findImplementation(.hemv, .complex_f64, .core_unit).?;
    try std.testing.expectEqual(contract.VectorStrideRule.unit, unit.strides.x);
    try std.testing.expectEqual(contract.VectorStrideRule.unit, unit.strides.y);
    try std.testing.expectEqual(Implementation.portable_scalar, unit.fallback.?.implementation);
    try std.testing.expect(unit.workspace.private_output);
    try std.testing.expect(unit.workspace.merge_required);
}
