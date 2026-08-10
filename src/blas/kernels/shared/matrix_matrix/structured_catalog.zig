// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Structured BLAS Level 3 implementation contracts.
//!
//! This registry describes the retained scalar/task foundations and the
//! correctly tested research implementations that remain outside the default
//! import graph. Shape thresholds and task counts belong to tuning, not here.

const std = @import("std");
const contract = @import("../../contract.zig");

pub const StructuredOperation = enum {
    syrk,
    herk,
    syr2k,
    her2k,
    symm,
    hemm,
    trmm,
    trsm,
};

pub const StructuredKernelId = enum {
    ssyrk_serial,
    dsyrk_serial,
    csyrk_serial,
    zsyrk_serial,
    cherk_serial,
    zherk_serial,
    ssyr2k_serial,
    dsyr2k_serial,
    csyr2k_serial,
    zsyr2k_serial,
    cher2k_serial,
    zher2k_serial,
    ssymm_serial,
    dsymm_serial,
    csymm_serial,
    zsymm_serial,
    chemm_serial,
    zhemm_serial,
    strmm_serial,
    dtrmm_serial,
    ctrmm_serial,
    ztrmm_serial,
    strsm_serial,
    dtrsm_serial,
    ctrsm_serial,
    ztrsm_serial,

    ssyrk_column_parallel,
    dsyrk_column_parallel,
    csyrk_column_parallel,
    zsyrk_column_parallel,
    cherk_column_parallel,
    zherk_column_parallel,
    ssyr2k_column_parallel,
    dsyr2k_column_parallel,
    csyr2k_column_parallel,
    zsyr2k_column_parallel,
    cher2k_column_parallel,
    zher2k_column_parallel,
    ssymm_column_parallel,
    dsymm_column_parallel,
    csymm_column_parallel,
    zsymm_column_parallel,
    chemm_column_parallel,
    zhemm_column_parallel,
    strmm_left_column_parallel,
    dtrmm_left_column_parallel,
    ctrmm_left_column_parallel,
    ztrmm_left_column_parallel,
    strsm_left_column_parallel,
    dtrsm_left_column_parallel,
    ctrsm_left_column_parallel,
    ztrsm_left_column_parallel,

    ssymm_dense_gemm_rejected,
    dsymm_dense_gemm_rejected,
    csymm_dense_gemm_rejected,
    zsymm_dense_gemm_rejected,
    chemm_dense_gemm_rejected,
    zhemm_dense_gemm_rejected,
    strmm_right_row_parallel_rejected,
    dtrmm_right_row_parallel_rejected,
    ctrmm_right_row_parallel_rejected,
    ztrmm_right_row_parallel_rejected,
    strsm_right_row_parallel_rejected,
    dtrsm_right_row_parallel_rejected,
    ctrsm_right_row_parallel_rejected,
    ztrsm_right_row_parallel_rejected,

    ssymm_dense_gemm_isolated,
    dsymm_dense_gemm_isolated,
    csymm_dense_gemm_isolated,
    zsymm_dense_gemm_isolated,
    chemm_dense_gemm_isolated,
    zhemm_dense_gemm_isolated,
    strmm_right_row_parallel_isolated,
    dtrmm_right_row_parallel_isolated,
    ctrmm_right_row_parallel_isolated,
    ztrmm_right_row_parallel_isolated,
    strsm_right_row_parallel_isolated,
    dtrsm_right_row_parallel_isolated,
    ctrsm_right_row_parallel_isolated,
    ztrsm_right_row_parallel_isolated,

    ssyrk_blocked,
    dsyrk_blocked,
    csyrk_blocked,
    zsyrk_blocked,
    cherk_blocked,
    zherk_blocked,
    ssyr2k_blocked,
    dsyr2k_blocked,
    csyr2k_blocked,
    zsyr2k_blocked,
    cher2k_blocked,
    zher2k_blocked,
    ssymm_blocked,
    dsymm_blocked,
    csymm_blocked,
    zsymm_blocked,
    chemm_blocked,
    zhemm_blocked,
    strmm_left_blocked,
    dtrmm_left_blocked,
    ctrmm_left_blocked,
    ztrmm_left_blocked,
    strmm_right_blocked,
    dtrmm_right_blocked,
    ctrmm_right_blocked,
    ztrmm_right_blocked,
    strsm_left_blocked,
    dtrsm_left_blocked,
    ctrsm_left_blocked,
    ztrsm_left_blocked,
    strsm_right_blocked,
    dtrsm_right_blocked,
    ctrsm_right_blocked,
    ztrsm_right_blocked,
};

pub const Implementation = enum {
    serial,
    retained_column_parallel,
    retained_left_column_parallel,
    rejected_dense_gemm,
    rejected_right_row_parallel,
    isolated_dense_gemm,
    isolated_right_row_parallel,
    blocked_rank_update,
    blocked_symmetric_multiply,
    blocked_triangular_left,
    blocked_triangular_right,
};

pub const StructuredKind = enum {
    symmetric,
    hermitian,
    triangular,
};

pub const SideSupport = packed struct {
    left: bool = false,
    right: bool = false,
};

pub const TransposeSupport = packed struct {
    no_trans: bool = true,
    trans: bool = false,
    conj_trans: bool = false,
};

pub const DiagonalSupport = packed struct {
    not_applicable: bool = true,
    unit: bool = false,
    non_unit: bool = false,
};

pub const ScalarOperand = enum {
    not_present,
    data_scalar,
    real_component,
};

pub const PackingKind = enum {
    none,
    dense_structured_materialization,
    structured_block_panels,
    private_output_tiles,
};

pub const WorkspaceFormula = enum {
    none,
    dense_order_squared_plus_optional_output,
    one_block,
    two_blocks,
};

pub const TaskTopology = enum {
    serial,
    cyclic_stored_columns,
    cyclic_output_columns,
    contiguous_output_columns,
    contiguous_output_rows,
    gemm_owned,
    gemm_blocked,
};

pub const Entrypoint = enum {
    symmetric_serial,
    symmetric_column_parallel,
    triangular_serial,
    triangular_left_column_parallel,
    symmetric_dense_gemm_rejected,
    triangular_right_parallel_rejected,
    symmetric_dense_gemm_isolated,
    triangular_right_parallel_isolated,
    structured_blocked_experimental,
};

pub const Descriptor = struct {
    name: []const u8,
    kernel: StructuredKernelId,
    fallback: ?StructuredKernelId,
    operation_family: contract.OperationKind,
    operation: StructuredOperation,
    scalar: contract.ScalarKind,
    capability: contract.IsaCapability,
    lifecycle: contract.Lifecycle,
    state: contract.StateContract,
    implementation: Implementation,
    entrypoint: Entrypoint,
    structure: StructuredKind,
    sides: SideSupport,
    transposes: TransposeSupport,
    diagonals: DiagonalSupport,
    structured_storage: contract.MatrixStorageKind,
    stored_window: contract.StoredWindow,
    output: contract.OutputOwnership,
    alpha: ScalarOperand,
    beta: ScalarOperand,
    tails: contract.TailSupport,
    packing: PackingKind,
    workspace_formula: WorkspaceFormula,
    max_workspace_bytes: usize,
    block_size: usize,
    workspace: contract.WorkspaceContract,
    traversal: contract.TraversalDependency,
    task_topology: TaskTopology,
    task_fallback: contract.TaskFallbackContract,
    fallback_atomicity: contract.FallbackAtomicity,
    fallback_semantics: contract.FallbackSemantics,
};

pub const registry = [_]Descriptor{
    make(.ssyrk_serial, null, .syrk, .f32, .serial),
    make(.dsyrk_serial, null, .syrk, .f64, .serial),
    make(.csyrk_serial, null, .syrk, .complex_f32, .serial),
    make(.zsyrk_serial, null, .syrk, .complex_f64, .serial),
    make(.cherk_serial, null, .herk, .complex_f32, .serial),
    make(.zherk_serial, null, .herk, .complex_f64, .serial),
    make(.ssyr2k_serial, null, .syr2k, .f32, .serial),
    make(.dsyr2k_serial, null, .syr2k, .f64, .serial),
    make(.csyr2k_serial, null, .syr2k, .complex_f32, .serial),
    make(.zsyr2k_serial, null, .syr2k, .complex_f64, .serial),
    make(.cher2k_serial, null, .her2k, .complex_f32, .serial),
    make(.zher2k_serial, null, .her2k, .complex_f64, .serial),
    make(.ssymm_serial, null, .symm, .f32, .serial),
    make(.dsymm_serial, null, .symm, .f64, .serial),
    make(.csymm_serial, null, .symm, .complex_f32, .serial),
    make(.zsymm_serial, null, .symm, .complex_f64, .serial),
    make(.chemm_serial, null, .hemm, .complex_f32, .serial),
    make(.zhemm_serial, null, .hemm, .complex_f64, .serial),
    make(.strmm_serial, null, .trmm, .f32, .serial),
    make(.dtrmm_serial, null, .trmm, .f64, .serial),
    make(.ctrmm_serial, null, .trmm, .complex_f32, .serial),
    make(.ztrmm_serial, null, .trmm, .complex_f64, .serial),
    make(.strsm_serial, null, .trsm, .f32, .serial),
    make(.dtrsm_serial, null, .trsm, .f64, .serial),
    make(.ctrsm_serial, null, .trsm, .complex_f32, .serial),
    make(.ztrsm_serial, null, .trsm, .complex_f64, .serial),

    make(.ssyrk_column_parallel, .ssyrk_serial, .syrk, .f32, .retained_column_parallel),
    make(.dsyrk_column_parallel, .dsyrk_serial, .syrk, .f64, .retained_column_parallel),
    make(.csyrk_column_parallel, .csyrk_serial, .syrk, .complex_f32, .retained_column_parallel),
    make(.zsyrk_column_parallel, .zsyrk_serial, .syrk, .complex_f64, .retained_column_parallel),
    make(.cherk_column_parallel, .cherk_serial, .herk, .complex_f32, .retained_column_parallel),
    make(.zherk_column_parallel, .zherk_serial, .herk, .complex_f64, .retained_column_parallel),
    make(.ssyr2k_column_parallel, .ssyr2k_serial, .syr2k, .f32, .retained_column_parallel),
    make(.dsyr2k_column_parallel, .dsyr2k_serial, .syr2k, .f64, .retained_column_parallel),
    make(.csyr2k_column_parallel, .csyr2k_serial, .syr2k, .complex_f32, .retained_column_parallel),
    make(.zsyr2k_column_parallel, .zsyr2k_serial, .syr2k, .complex_f64, .retained_column_parallel),
    make(.cher2k_column_parallel, .cher2k_serial, .her2k, .complex_f32, .retained_column_parallel),
    make(.zher2k_column_parallel, .zher2k_serial, .her2k, .complex_f64, .retained_column_parallel),
    make(.ssymm_column_parallel, .ssymm_serial, .symm, .f32, .retained_column_parallel),
    make(.dsymm_column_parallel, .dsymm_serial, .symm, .f64, .retained_column_parallel),
    make(.csymm_column_parallel, .csymm_serial, .symm, .complex_f32, .retained_column_parallel),
    make(.zsymm_column_parallel, .zsymm_serial, .symm, .complex_f64, .retained_column_parallel),
    make(.chemm_column_parallel, .chemm_serial, .hemm, .complex_f32, .retained_column_parallel),
    make(.zhemm_column_parallel, .zhemm_serial, .hemm, .complex_f64, .retained_column_parallel),
    make(.strmm_left_column_parallel, .strmm_serial, .trmm, .f32, .retained_left_column_parallel),
    make(.dtrmm_left_column_parallel, .dtrmm_serial, .trmm, .f64, .retained_left_column_parallel),
    make(.ctrmm_left_column_parallel, .ctrmm_serial, .trmm, .complex_f32, .retained_left_column_parallel),
    make(.ztrmm_left_column_parallel, .ztrmm_serial, .trmm, .complex_f64, .retained_left_column_parallel),
    make(.strsm_left_column_parallel, .strsm_serial, .trsm, .f32, .retained_left_column_parallel),
    make(.dtrsm_left_column_parallel, .dtrsm_serial, .trsm, .f64, .retained_left_column_parallel),
    make(.ctrsm_left_column_parallel, .ctrsm_serial, .trsm, .complex_f32, .retained_left_column_parallel),
    make(.ztrsm_left_column_parallel, .ztrsm_serial, .trsm, .complex_f64, .retained_left_column_parallel),

    make(.ssymm_dense_gemm_rejected, .ssymm_column_parallel, .symm, .f32, .rejected_dense_gemm),
    make(.dsymm_dense_gemm_rejected, .dsymm_column_parallel, .symm, .f64, .rejected_dense_gemm),
    make(.csymm_dense_gemm_rejected, .csymm_column_parallel, .symm, .complex_f32, .rejected_dense_gemm),
    make(.zsymm_dense_gemm_rejected, .zsymm_column_parallel, .symm, .complex_f64, .rejected_dense_gemm),
    make(.chemm_dense_gemm_rejected, .chemm_column_parallel, .hemm, .complex_f32, .rejected_dense_gemm),
    make(.zhemm_dense_gemm_rejected, .zhemm_column_parallel, .hemm, .complex_f64, .rejected_dense_gemm),
    make(.strmm_right_row_parallel_rejected, .strmm_serial, .trmm, .f32, .rejected_right_row_parallel),
    make(.dtrmm_right_row_parallel_rejected, .dtrmm_serial, .trmm, .f64, .rejected_right_row_parallel),
    make(.ctrmm_right_row_parallel_rejected, .ctrmm_serial, .trmm, .complex_f32, .rejected_right_row_parallel),
    make(.ztrmm_right_row_parallel_rejected, .ztrmm_serial, .trmm, .complex_f64, .rejected_right_row_parallel),
    make(.strsm_right_row_parallel_rejected, .strsm_serial, .trsm, .f32, .rejected_right_row_parallel),
    make(.dtrsm_right_row_parallel_rejected, .dtrsm_serial, .trsm, .f64, .rejected_right_row_parallel),
    make(.ctrsm_right_row_parallel_rejected, .ctrsm_serial, .trsm, .complex_f32, .rejected_right_row_parallel),
    make(.ztrsm_right_row_parallel_rejected, .ztrsm_serial, .trsm, .complex_f64, .rejected_right_row_parallel),

    // These IDs describe the link-isolated retry. They intentionally remain
    // distinct from the earlier in-graph experiments above so evidence and
    // lifecycle decisions cannot leak across the isolation boundary.
    make(.ssymm_dense_gemm_isolated, .ssymm_column_parallel, .symm, .f32, .isolated_dense_gemm),
    make(.dsymm_dense_gemm_isolated, .dsymm_column_parallel, .symm, .f64, .isolated_dense_gemm),
    make(.csymm_dense_gemm_isolated, .csymm_column_parallel, .symm, .complex_f32, .isolated_dense_gemm),
    make(.zsymm_dense_gemm_isolated, .zsymm_column_parallel, .symm, .complex_f64, .isolated_dense_gemm),
    make(.chemm_dense_gemm_isolated, .chemm_column_parallel, .hemm, .complex_f32, .isolated_dense_gemm),
    make(.zhemm_dense_gemm_isolated, .zhemm_column_parallel, .hemm, .complex_f64, .isolated_dense_gemm),
    make(.strmm_right_row_parallel_isolated, .strmm_serial, .trmm, .f32, .isolated_right_row_parallel),
    make(.dtrmm_right_row_parallel_isolated, .dtrmm_serial, .trmm, .f64, .isolated_right_row_parallel),
    make(.ctrmm_right_row_parallel_isolated, .ctrmm_serial, .trmm, .complex_f32, .isolated_right_row_parallel),
    make(.ztrmm_right_row_parallel_isolated, .ztrmm_serial, .trmm, .complex_f64, .isolated_right_row_parallel),
    make(.strsm_right_row_parallel_isolated, .strsm_serial, .trsm, .f32, .isolated_right_row_parallel),
    make(.dtrsm_right_row_parallel_isolated, .dtrsm_serial, .trsm, .f64, .isolated_right_row_parallel),
    make(.ctrsm_right_row_parallel_isolated, .ctrsm_serial, .trsm, .complex_f32, .isolated_right_row_parallel),
    make(.ztrsm_right_row_parallel_isolated, .ztrsm_serial, .trsm, .complex_f64, .isolated_right_row_parallel),

    make(.ssyrk_blocked, .ssyrk_column_parallel, .syrk, .f32, .blocked_rank_update),
    make(.dsyrk_blocked, .dsyrk_column_parallel, .syrk, .f64, .blocked_rank_update),
    make(.csyrk_blocked, .csyrk_column_parallel, .syrk, .complex_f32, .blocked_rank_update),
    make(.zsyrk_blocked, .zsyrk_column_parallel, .syrk, .complex_f64, .blocked_rank_update),
    make(.cherk_blocked, .cherk_column_parallel, .herk, .complex_f32, .blocked_rank_update),
    make(.zherk_blocked, .zherk_column_parallel, .herk, .complex_f64, .blocked_rank_update),
    make(.ssyr2k_blocked, .ssyr2k_column_parallel, .syr2k, .f32, .blocked_rank_update),
    make(.dsyr2k_blocked, .dsyr2k_column_parallel, .syr2k, .f64, .blocked_rank_update),
    make(.csyr2k_blocked, .csyr2k_column_parallel, .syr2k, .complex_f32, .blocked_rank_update),
    make(.zsyr2k_blocked, .zsyr2k_column_parallel, .syr2k, .complex_f64, .blocked_rank_update),
    make(.cher2k_blocked, .cher2k_column_parallel, .her2k, .complex_f32, .blocked_rank_update),
    make(.zher2k_blocked, .zher2k_column_parallel, .her2k, .complex_f64, .blocked_rank_update),
    make(.ssymm_blocked, .ssymm_column_parallel, .symm, .f32, .blocked_symmetric_multiply),
    make(.dsymm_blocked, .dsymm_column_parallel, .symm, .f64, .blocked_symmetric_multiply),
    make(.csymm_blocked, .csymm_column_parallel, .symm, .complex_f32, .blocked_symmetric_multiply),
    make(.zsymm_blocked, .zsymm_column_parallel, .symm, .complex_f64, .blocked_symmetric_multiply),
    make(.chemm_blocked, .chemm_column_parallel, .hemm, .complex_f32, .blocked_symmetric_multiply),
    make(.zhemm_blocked, .zhemm_column_parallel, .hemm, .complex_f64, .blocked_symmetric_multiply),
    make(.strmm_left_blocked, .strmm_left_column_parallel, .trmm, .f32, .blocked_triangular_left),
    make(.dtrmm_left_blocked, .dtrmm_left_column_parallel, .trmm, .f64, .blocked_triangular_left),
    make(.ctrmm_left_blocked, .ctrmm_left_column_parallel, .trmm, .complex_f32, .blocked_triangular_left),
    make(.ztrmm_left_blocked, .ztrmm_left_column_parallel, .trmm, .complex_f64, .blocked_triangular_left),
    make(.strmm_right_blocked, .strmm_serial, .trmm, .f32, .blocked_triangular_right),
    make(.dtrmm_right_blocked, .dtrmm_serial, .trmm, .f64, .blocked_triangular_right),
    make(.ctrmm_right_blocked, .ctrmm_serial, .trmm, .complex_f32, .blocked_triangular_right),
    make(.ztrmm_right_blocked, .ztrmm_serial, .trmm, .complex_f64, .blocked_triangular_right),
    make(.strsm_left_blocked, .strsm_left_column_parallel, .trsm, .f32, .blocked_triangular_left),
    make(.dtrsm_left_blocked, .dtrsm_left_column_parallel, .trsm, .f64, .blocked_triangular_left),
    make(.ctrsm_left_blocked, .ctrsm_left_column_parallel, .trsm, .complex_f32, .blocked_triangular_left),
    make(.ztrsm_left_blocked, .ztrsm_left_column_parallel, .trsm, .complex_f64, .blocked_triangular_left),
    make(.strsm_right_blocked, .strsm_serial, .trsm, .f32, .blocked_triangular_right),
    make(.dtrsm_right_blocked, .dtrsm_serial, .trsm, .f64, .blocked_triangular_right),
    make(.ctrsm_right_blocked, .ctrsm_serial, .trsm, .complex_f32, .blocked_triangular_right),
    make(.ztrsm_right_blocked, .ztrsm_serial, .trsm, .complex_f64, .blocked_triangular_right),
};

fn make(comptime kernel: StructuredKernelId, comptime fallback: ?StructuredKernelId, comptime operation: StructuredOperation, comptime scalar: contract.ScalarKind, comptime implementation: Implementation) Descriptor {
    const serial = implementation == .serial;
    const rank = operation == .syrk or operation == .herk or operation == .syr2k or operation == .her2k;
    const triangular = operation == .trmm or operation == .trsm;
    const hermitian = operation == .herk or operation == .her2k or operation == .hemm;
    const task_parallel = implementation == .retained_column_parallel or
        implementation == .retained_left_column_parallel or
        implementation == .rejected_right_row_parallel or
        implementation == .isolated_right_row_parallel;
    const dense_gemm = implementation == .rejected_dense_gemm or implementation == .isolated_dense_gemm;
    const blocked_rank = implementation == .blocked_rank_update;
    const blocked_symmetric = implementation == .blocked_symmetric_multiply;
    const blocked_triangular = implementation == .blocked_triangular_left or implementation == .blocked_triangular_right;
    const blocked = blocked_rank or blocked_symmetric or blocked_triangular;
    const blocked_buffers: usize = if (operation == .trmm and blocked_triangular) 2 else 1;

    return .{
        .name = @tagName(kernel),
        .kernel = kernel,
        .fallback = fallback,
        .operation_family = .structured_matrix_matrix,
        .operation = operation,
        .scalar = scalar,
        .capability = if (implementation == .rejected_dense_gemm or
            implementation == .rejected_right_row_parallel or
            implementation == .isolated_dense_gemm or
            implementation == .isolated_right_row_parallel)
            .x86_64_sse2
        else
            .generic,
        .lifecycle = if (serial)
            .portable_fallback
        else if (implementation == .rejected_dense_gemm or implementation == .rejected_right_row_parallel)
            .rejected
        else if (implementation == .isolated_dense_gemm or implementation == .isolated_right_row_parallel)
            .experimental
        else if (blocked)
            .experimental
        else
            .production,
        .state = .{},
        .implementation = implementation,
        .entrypoint = entrypointFor(operation, implementation),
        .structure = if (hermitian) .hermitian else if (triangular) .triangular else .symmetric,
        .sides = if (rank)
            .{}
        else if (implementation == .retained_left_column_parallel)
            .{ .left = true }
        else if (implementation == .blocked_triangular_left)
            .{ .left = true }
        else if (implementation == .rejected_right_row_parallel or implementation == .isolated_right_row_parallel)
            .{ .right = true }
        else if (implementation == .blocked_triangular_right)
            .{ .right = true }
        else
            .{ .left = true, .right = true },
        .transposes = transposeSupport(operation),
        .diagonals = if (triangular)
            .{ .not_applicable = false, .unit = true, .non_unit = true }
        else
            .{},
        .structured_storage = .dense_triangle,
        .stored_window = .selected_triangle,
        .output = if (rank)
            .stored_matrix_columns
        else if (triangular)
            .in_place_full_matrix
        else
            .full_general_matrix,
        .alpha = if (operation == .herk) .real_component else .data_scalar,
        .beta = if (operation == .trmm or operation == .trsm)
            .not_present
        else if (operation == .herk or operation == .her2k)
            .real_component
        else
            .data_scalar,
        .tails = .{ .m = .native, .n = .native, .k = .native },
        .packing = if (dense_gemm)
            .dense_structured_materialization
        else if (blocked_rank)
            .private_output_tiles
        else if (blocked)
            .structured_block_panels
        else
            .none,
        .workspace_formula = if (dense_gemm)
            .dense_order_squared_plus_optional_output
        else if (blocked_buffers == 2)
            .two_blocks
        else if (blocked)
            .one_block
        else
            .none,
        .max_workspace_bytes = if (dense_gemm)
            64 * 1024 * 1024
        else if (blocked)
            64 * 64 * scalarBytes(scalar) * blocked_buffers
        else
            0,
        .block_size = if (blocked) 64 else 0,
        .workspace = .{},
        .traversal = if (triangular) .triangular_ordered else .none,
        .task_topology = taskTopology(operation, implementation),
        .task_fallback = if (task_parallel) .disjoint_outputs_all_tasks_required else .not_applicable,
        .fallback_atomicity = .reject_before_write,
        .fallback_semantics = if (serial) .terminal_portable else .whole_operation,
    };
}

fn entrypointFor(operation: StructuredOperation, implementation: Implementation) Entrypoint {
    return switch (implementation) {
        .serial => if (operation == .trmm or operation == .trsm) .triangular_serial else .symmetric_serial,
        .retained_column_parallel => .symmetric_column_parallel,
        .retained_left_column_parallel => .triangular_left_column_parallel,
        .rejected_dense_gemm => .symmetric_dense_gemm_rejected,
        .rejected_right_row_parallel => .triangular_right_parallel_rejected,
        .isolated_dense_gemm => .symmetric_dense_gemm_isolated,
        .isolated_right_row_parallel => .triangular_right_parallel_isolated,
        .blocked_rank_update, .blocked_symmetric_multiply, .blocked_triangular_left, .blocked_triangular_right => .structured_blocked_experimental,
    };
}

fn transposeSupport(operation: StructuredOperation) TransposeSupport {
    return switch (operation) {
        .syrk, .syr2k => .{ .trans = true },
        .herk, .her2k => .{ .conj_trans = true },
        .symm, .hemm => .{},
        .trmm, .trsm => .{ .trans = true, .conj_trans = true },
    };
}

fn taskTopology(operation: StructuredOperation, implementation: Implementation) TaskTopology {
    return switch (implementation) {
        .serial => .serial,
        .retained_column_parallel => if (operation == .symm or operation == .hemm) .cyclic_output_columns else .cyclic_stored_columns,
        .retained_left_column_parallel => .contiguous_output_columns,
        .rejected_dense_gemm => .gemm_owned,
        .rejected_right_row_parallel => .contiguous_output_rows,
        .isolated_dense_gemm => .gemm_owned,
        .isolated_right_row_parallel => .contiguous_output_rows,
        .blocked_rank_update, .blocked_symmetric_multiply, .blocked_triangular_left, .blocked_triangular_right => .gemm_blocked,
    };
}

fn scalarBytes(scalar: contract.ScalarKind) usize {
    return switch (scalar) {
        .f32 => 4,
        .f64 => 8,
        .complex_f32 => 8,
        .complex_f64 => 16,
    };
}

fn operationSupportsScalar(operation: StructuredOperation, scalar: contract.ScalarKind) bool {
    return switch (operation) {
        .herk, .her2k, .hemm => scalar == .complex_f32 or scalar == .complex_f64,
        else => true,
    };
}

fn validateRegistry() void {
    for (registry, 0..) |descriptor, index| {
        if (descriptor.operation_family != .structured_matrix_matrix) @compileError("structured descriptor has wrong operation family");
        if (!operationSupportsScalar(descriptor.operation, descriptor.scalar)) @compileError("structured descriptor has illegal scalar");
        if (descriptor.name.len == 0) @compileError("structured descriptor name is empty");
        if (descriptor.structured_storage != .dense_triangle or descriptor.stored_window != .selected_triangle) {
            @compileError("structured dense descriptor lost selected-triangle input semantics");
        }

        const serial = descriptor.implementation == .serial;
        if (serial) {
            if (descriptor.lifecycle != .portable_fallback or descriptor.fallback != null or descriptor.fallback_semantics != .terminal_portable) {
                @compileError("serial structured fallback is not terminal");
            }
        } else {
            if (descriptor.fallback == null or descriptor.fallback_semantics != .whole_operation) {
                @compileError("structured implementation lacks whole-operation fallback");
            }
            const fallback = descriptorForKernel(descriptor.fallback.?) orelse @compileError("structured fallback is absent");
            if (fallback.operation != descriptor.operation or fallback.scalar != descriptor.scalar) {
                @compileError("structured fallback changes operation or scalar");
            }
        }

        const rank = descriptor.operation == .syrk or descriptor.operation == .herk or descriptor.operation == .syr2k or descriptor.operation == .her2k;
        const triangular = descriptor.operation == .trmm or descriptor.operation == .trsm;
        if (rank and descriptor.output != .stored_matrix_columns) @compileError("rank-k descriptor must own only stored columns");
        if (triangular and (descriptor.output != .in_place_full_matrix or descriptor.traversal != .triangular_ordered)) {
            @compileError("triangular matrix descriptor lost in-place ordered ownership");
        }
        if (!rank and !triangular and descriptor.output != .full_general_matrix) @compileError("symmetric multiply must own full C");

        const has_workspace = descriptor.workspace_formula != .none;
        if (has_workspace != (descriptor.packing != .none) or has_workspace != (descriptor.max_workspace_bytes != 0)) {
            @compileError("structured packing and workspace contracts disagree");
        }
        const blocked = descriptor.implementation == .blocked_rank_update or
            descriptor.implementation == .blocked_symmetric_multiply or
            descriptor.implementation == .blocked_triangular_left or
            descriptor.implementation == .blocked_triangular_right;
        if (blocked != (descriptor.block_size != 0)) @compileError("blocked structured descriptor lost block geometry");
        if (blocked and descriptor.lifecycle != .experimental) @compileError("unbenchmarked blocked structured path cannot be default eligible");
        if (descriptor.implementation == .rejected_dense_gemm and descriptor.lifecycle != .rejected) {
            @compileError("dense-GEMM structured experiment cannot be default eligible");
        }
        if (descriptor.implementation == .rejected_right_row_parallel and descriptor.lifecycle != .rejected) {
            @compileError("right-row triangular experiment cannot be default eligible");
        }
        if ((descriptor.implementation == .isolated_dense_gemm or
            descriptor.implementation == .isolated_right_row_parallel) and
            descriptor.lifecycle != .experimental)
        {
            @compileError("link-isolated structured retry must remain experimental");
        }
        var current = descriptor;
        var depth: usize = 0;
        while (current.fallback) |fallback_id| : (depth += 1) {
            if (depth >= registry.len) @compileError("cyclic structured fallback chain");
            current = descriptorForKernel(fallback_id) orelse @compileError("structured fallback is absent");
        }
        if (current.lifecycle != .portable_fallback or current.fallback_semantics != .terminal_portable) {
            @compileError("structured fallback chain is not total");
        }
        for (registry[index + 1 ..]) |other| {
            if (descriptor.kernel == other.kernel) @compileError("duplicate structured kernel ID");
            if (std.mem.eql(u8, descriptor.name, other.name)) @compileError("duplicate structured kernel name");
        }
    }
}

comptime {
    @setEvalBranchQuota(30_000);
    validateRegistry();
}

pub fn descriptorForKernel(kernel: StructuredKernelId) ?Descriptor {
    for (registry) |descriptor| if (descriptor.kernel == kernel) return descriptor;
    return null;
}

test "structured Level 3 registry preserves storage output and lifecycle boundaries" {
    try std.testing.expectEqual(@as(usize, 114), registry.len);
    var default_eligible: usize = 0;
    var rejected: usize = 0;
    var experimental: usize = 0;
    for (registry) |descriptor| {
        default_eligible += @intFromBool(descriptor.lifecycle.defaultEligible());
        rejected += @intFromBool(descriptor.lifecycle == .rejected);
        experimental += @intFromBool(descriptor.lifecycle == .experimental);
    }
    try std.testing.expectEqual(@as(usize, 52), default_eligible);
    try std.testing.expectEqual(@as(usize, 14), rejected);
    try std.testing.expectEqual(@as(usize, 48), experimental);

    const herk = descriptorForKernel(.cherk_column_parallel).?;
    try std.testing.expectEqual(StructuredKind.hermitian, herk.structure);
    try std.testing.expect(herk.transposes.no_trans and herk.transposes.conj_trans and !herk.transposes.trans);
    try std.testing.expectEqual(ScalarOperand.real_component, herk.alpha);
    try std.testing.expectEqual(ScalarOperand.real_component, herk.beta);
    try std.testing.expectEqual(contract.OutputOwnership.stored_matrix_columns, herk.output);

    const left_trsm = descriptorForKernel(.ztrsm_left_column_parallel).?;
    try std.testing.expect(left_trsm.sides.left and !left_trsm.sides.right);
    try std.testing.expectEqual(contract.OutputOwnership.in_place_full_matrix, left_trsm.output);
    try std.testing.expectEqual(contract.TraversalDependency.triangular_ordered, left_trsm.traversal);
    try std.testing.expectEqual(contract.TaskFallbackContract.disjoint_outputs_all_tasks_required, left_trsm.task_fallback);

    const dense_symm = descriptorForKernel(.ssymm_dense_gemm_rejected).?;
    try std.testing.expectEqual(contract.Lifecycle.rejected, dense_symm.lifecycle);
    try std.testing.expectEqual(PackingKind.dense_structured_materialization, dense_symm.packing);
    try std.testing.expectEqual(@as(usize, 64 * 1024 * 1024), dense_symm.max_workspace_bytes);
    try std.testing.expectEqual(StructuredKernelId.ssymm_column_parallel, dense_symm.fallback.?);

    const right_trmm = descriptorForKernel(.strmm_right_row_parallel_rejected).?;
    try std.testing.expect(!right_trmm.sides.left and right_trmm.sides.right);
    try std.testing.expectEqual(contract.Lifecycle.rejected, right_trmm.lifecycle);
    try std.testing.expectEqual(StructuredKernelId.strmm_serial, right_trmm.fallback.?);

    const isolated_dense = descriptorForKernel(.zsymm_dense_gemm_isolated).?;
    try std.testing.expectEqual(contract.Lifecycle.experimental, isolated_dense.lifecycle);
    try std.testing.expectEqual(Entrypoint.symmetric_dense_gemm_isolated, isolated_dense.entrypoint);
    try std.testing.expectEqual(PackingKind.dense_structured_materialization, isolated_dense.packing);

    const isolated_right = descriptorForKernel(.ztrsm_right_row_parallel_isolated).?;
    try std.testing.expectEqual(contract.Lifecycle.experimental, isolated_right.lifecycle);
    try std.testing.expect(!isolated_right.sides.left and isolated_right.sides.right);
    try std.testing.expectEqual(Entrypoint.triangular_right_parallel_isolated, isolated_right.entrypoint);

    const blocked_her2k = descriptorForKernel(.zher2k_blocked).?;
    try std.testing.expectEqual(contract.Lifecycle.experimental, blocked_her2k.lifecycle);
    try std.testing.expectEqual(PackingKind.private_output_tiles, blocked_her2k.packing);
    try std.testing.expectEqual(@as(usize, 64), blocked_her2k.block_size);
    try std.testing.expectEqual(@as(usize, 64 * 64 * 16), blocked_her2k.max_workspace_bytes);

    const blocked_right_trsm = descriptorForKernel(.dtrsm_right_blocked).?;
    try std.testing.expect(!blocked_right_trsm.sides.left and blocked_right_trsm.sides.right);
    try std.testing.expectEqual(StructuredKernelId.dtrsm_serial, blocked_right_trsm.fallback.?);
}
