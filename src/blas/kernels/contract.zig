// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Common metadata shared by kernel registries.
//!
//! These types describe executable facts. Shape preferences, measured cutoffs,
//! and thread-count policy belong in tuning modules rather than this contract.

const std = @import("std");
const types = @import("../types.zig");

pub const OperationKind = enum {
    scalar_generator,
    vector_unary,
    vector_binary,
    matrix_vector,
    matrix_rank_update,
    matrix_matrix,
    structured_matrix_matrix,
};

pub const ScalarKind = enum {
    f32,
    f64,
    complex_f32,
    complex_f64,
};

pub const IsaCapability = enum {
    generic,
    aarch64_asimd_fma,
    aarch64_sve,
    aarch64_sve2,
    aarch64_sme,
    aarch64_sme2,
    apple_amx,
    x86_64_sse2,
    x86_64_avx,
    x86_64_avx2,
    x86_64_fma,
    x86_64_avx2_fma,
    x86_64_avx512f,
    x86_64_avx512f_fma,
    x86_64_amx,
};

pub const Lifecycle = enum {
    portable_fallback,
    production,
    experimental,
    rejected,

    pub fn defaultEligible(self: Lifecycle) bool {
        return self == .portable_fallback or self == .production;
    }
};

pub const TailStrategy = enum {
    native,
    predicated,
    scalar_cleanup,
    whole_operation_fallback,
};

pub const TailSupport = struct {
    m: TailStrategy,
    n: TailStrategy,
    k: TailStrategy,
};

/// A hard stride precondition. Performance cutoffs do not belong here.
pub const VectorStrideRule = enum {
    not_applicable,
    any_nonzero,
    positive,
    unit,
    exactly_two,
};

pub const VectorStrideContract = struct {
    x: VectorStrideRule,
    y: VectorStrideRule = .not_applicable,
};

/// Required byte alignment at the implementation boundary. A value of one
/// means the implementation deliberately uses unaligned-safe accesses.
pub const AlignmentContract = struct {
    x_bytes: usize = 1,
    y_bytes: usize = 1,
};

pub const AliasContract = enum {
    /// All alias relationships permitted by the public BLAS operation are
    /// handled by this execution layer, possibly by falling back internally.
    blas_valid,
    non_overlapping,
};

pub const ReductionKind = enum {
    none,
    sum,
    scaled_sum_of_squares,
    first_index_maximum,
};

pub const ReductionContract = struct {
    kind: ReductionKind = .none,
    preserves_first_index_on_ties: bool = false,
};

pub const FallbackSemantics = enum {
    terminal_portable,
    whole_operation,
};

pub const LayoutSupport = packed struct {
    no_trans: bool = true,
    transposed_a: bool = false,
    transposed_b: bool = false,
    conjugated_a: bool = false,
    conjugated_b: bool = false,
};

pub const EpilogueSupport = packed struct {
    arbitrary_alpha: bool = true,
    arbitrary_beta: bool = true,
    alpha_zero: bool = true,
    beta_zero: bool = true,
};

pub const StateKind = enum {
    none,
    aarch64_streaming_sm,
    aarch64_streaming_za,
    apple_amx,
    x86_64_amx,
};

pub const StateBoundary = enum {
    not_applicable,
    disabled,
};

pub const StateTransition = enum {
    none,
    aarch64_smstart_sm_smstop_sm,
    aarch64_smstart_sm_za_smstop_za_sm,
    apple_amx_set_clear,
    x86_64_ldtilecfg_tilerelease,
};

pub const StateClobbers = packed struct {
    scalable_vector_registers: bool = false,
    predicate_registers: bool = false,
    first_fault_register: bool = false,
    matrix_accumulator: bool = false,
    tile_configuration: bool = false,
    memory: bool = false,
};

pub const StateContract = struct {
    kind: StateKind = .none,
    compiler_support_required: bool = false,
    os_permission_required: bool = false,
    implementation_owns_entry_exit: bool = false,
    entry: StateBoundary = .not_applicable,
    exit: StateBoundary = .not_applicable,
    transition: StateTransition = .none,
    cleanup_on_success: bool = false,
    cleanup_on_failure: bool = false,
    clobbers: StateClobbers = .{},
};

pub fn stateContract(kind: StateKind) StateContract {
    return switch (kind) {
        .none => .{},
        .aarch64_streaming_sm => .{
            .kind = kind,
            .compiler_support_required = true,
            .os_permission_required = true,
            .implementation_owns_entry_exit = true,
            .entry = .disabled,
            .exit = .disabled,
            .transition = .aarch64_smstart_sm_smstop_sm,
            .cleanup_on_success = true,
            .cleanup_on_failure = true,
            .clobbers = .{
                .scalable_vector_registers = true,
                .predicate_registers = true,
                .first_fault_register = true,
                .memory = true,
            },
        },
        .aarch64_streaming_za => .{
            .kind = kind,
            .compiler_support_required = true,
            .os_permission_required = true,
            .implementation_owns_entry_exit = true,
            .entry = .disabled,
            .exit = .disabled,
            .transition = .aarch64_smstart_sm_za_smstop_za_sm,
            .cleanup_on_success = true,
            .cleanup_on_failure = true,
            .clobbers = .{
                .scalable_vector_registers = true,
                .predicate_registers = true,
                .first_fault_register = true,
                .matrix_accumulator = true,
                .memory = true,
            },
        },
        .apple_amx => .{
            .kind = kind,
            .compiler_support_required = true,
            .os_permission_required = true,
            .implementation_owns_entry_exit = true,
            .entry = .disabled,
            .exit = .disabled,
            .transition = .apple_amx_set_clear,
            .cleanup_on_success = true,
            .cleanup_on_failure = true,
            .clobbers = .{
                .matrix_accumulator = true,
                .memory = true,
            },
        },
        .x86_64_amx => .{
            .kind = kind,
            .compiler_support_required = true,
            .os_permission_required = true,
            .implementation_owns_entry_exit = true,
            .entry = .disabled,
            .exit = .disabled,
            .transition = .x86_64_ldtilecfg_tilerelease,
            .cleanup_on_success = true,
            .cleanup_on_failure = true,
            .clobbers = .{
                .matrix_accumulator = true,
                .tile_configuration = true,
                .memory = true,
            },
        },
    };
}

pub fn validateStateContract(comptime state: StateContract) void {
    @setEvalBranchQuota(5000);
    if (state.kind == .none) {
        if (!std.meta.eql(state, StateContract{})) @compileError("stateless kernel declares architectural state requirements");
        return;
    }
    if (!state.compiler_support_required or !state.os_permission_required or !state.implementation_owns_entry_exit) {
        @compileError("stateful kernel lacks compiler, OS, or ownership requirements");
    }
    if (state.entry != .disabled or state.exit != .disabled) @compileError("stateful internal kernel must enter and leave with its facility disabled");
    if (state.transition == .none) @compileError("stateful kernel lacks an entry/exit transition");
    if (!state.cleanup_on_success or !state.cleanup_on_failure) @compileError("stateful kernel lacks total cleanup semantics");
    const expected = stateContract(state.kind);
    if (!std.meta.eql(state, expected)) @compileError("state contract does not match its architectural state kind");
}

pub const MatrixStorageKind = enum {
    not_applicable,
    dense_general,
    dense_triangle,
    general_band,
    symmetric_band,
    triangular_band,
    packed_triangle,
};

pub const StoredWindow = enum {
    full_matrix,
    selected_triangle,
    exact_band,
    packed_columns,
};

pub const OutputOwnership = enum {
    final_vector,
    additive_vector_region,
    private_vector_delta,
    in_place_dependency_vector,
    full_general_matrix,
    in_place_full_matrix,
    stored_matrix_columns,
};

pub const FallbackAtomicity = enum {
    reject_before_write,
    completes_operation,
};

pub const WorkspaceContract = struct {
    private_output: bool = false,
    merge_required: bool = false,
    allocation_failure_falls_back_whole_operation: bool = true,
};

pub const TaskFallbackContract = enum {
    /// The implementation does not create a task composition of its own.
    not_applicable,
    /// Tasks write only private buffers; caller output is committed after every
    /// task completes, and a pre-submit failure may run the whole fallback.
    private_results_commit_after_all_tasks,
    /// Tasks own disjoint caller-output regions. Submission rejection occurs
    /// before the first task; after execution starts every task must complete.
    disjoint_outputs_all_tasks_required,
};

pub const TraversalDependency = enum {
    none,
    triangular_ordered,
};

pub fn scalarKind(comptime T: type) ScalarKind {
    return switch (T) {
        f32 => .f32,
        f64 => .f64,
        types.ComplexF32 => .complex_f32,
        types.ComplexF64 => .complex_f64,
        else => @compileError("unsupported kernel scalar type"),
    };
}

test "kernel lifecycle controls default eligibility" {
    try std.testing.expect(Lifecycle.portable_fallback.defaultEligible());
    try std.testing.expect(Lifecycle.production.defaultEligible());
    try std.testing.expect(!Lifecycle.experimental.defaultEligible());
    try std.testing.expect(!Lifecycle.rejected.defaultEligible());
}

test "real scalar kinds are stable" {
    try std.testing.expectEqual(ScalarKind.f32, scalarKind(f32));
    try std.testing.expectEqual(ScalarKind.f64, scalarKind(f64));
}

test "complex scalar kinds are stable" {
    try std.testing.expectEqual(ScalarKind.complex_f32, scalarKind(types.ComplexF32));
    try std.testing.expectEqual(ScalarKind.complex_f64, scalarKind(types.ComplexF64));
}

test "private Level 2 outputs declare a merge obligation" {
    const workspace: WorkspaceContract = .{
        .private_output = true,
        .merge_required = true,
    };
    try std.testing.expect(workspace.private_output);
    try std.testing.expect(workspace.merge_required);
    try std.testing.expect(workspace.allocation_failure_falls_back_whole_operation);
}

test "Level 2 task fallback distinguishes private commit from disjoint output" {
    try std.testing.expect(TaskFallbackContract.private_results_commit_after_all_tasks !=
        TaskFallbackContract.disjoint_outputs_all_tasks_required);
}

test "stateful ISA contracts declare entry exit clobbers and total cleanup" {
    inline for (.{
        StateKind.aarch64_streaming_sm,
        StateKind.aarch64_streaming_za,
        StateKind.apple_amx,
        StateKind.x86_64_amx,
    }) |kind| {
        const state = comptime stateContract(kind);
        comptime validateStateContract(state);
        try std.testing.expect(state.compiler_support_required);
        try std.testing.expect(state.os_permission_required);
        try std.testing.expect(state.implementation_owns_entry_exit);
        try std.testing.expectEqual(StateBoundary.disabled, state.entry);
        try std.testing.expectEqual(StateBoundary.disabled, state.exit);
        try std.testing.expect(state.cleanup_on_success);
        try std.testing.expect(state.cleanup_on_failure);
        try std.testing.expect(state.clobbers.memory);
    }
    comptime validateStateContract(.{});
}
