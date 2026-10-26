// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Optional synchronous phase agreement. This borrows caller communication;
//! it creates no communicator, allocator, execution owner or persistent handle.
//! Every participant uses the same entry and descriptor. The callback first
//! agrees any local failure; only when all succeed does it compare the exact
//! stage/index/control tuple. It returns the same failure on every participant.
//! A nullable operation is borrowed only during the callback (admission may fail).
const scalar = @import("../blas/core/shared/scalar.zig");
const execution = @import("zynum-execution");

pub const Stage = enum(u32) {
    entry = 1,
    scalar_kind,
    admission,
    workspace,
    vector_allocation,
    real_allocation,
    preflight,
    coefficients_count,
    coefficient_real,
    coefficient_imaginary,
    panel_columns,
    panel_backend,
    scale_alpha_real,
    scale_alpha_imaginary,
    scale_beta_real,
    scale_beta_imaginary,
    action_complete,
    local_arithmetic,
    initial_norm,
    unit_norm,
    global_dimension,
    max_order,
    reduction_backend,
    ritz_tolerance,
    hermitian_tolerance,
    breakdown_tolerance,
    ritz_budget_present,
    ritz_budget,
    projection_real,
    projection_imaginary,
    residual_norm,
    minimum_ritz,
    maximum_ritz,
    minimum_change,
    maximum_change,
    stopping_decision,
    commit,
    preconditioner_backend,
    initial_backend,
    relative_tolerance,
    absolute_tolerance,
    iteration_limit,
    preconditioner_complete,
    residual_restart,
    rational_node,
    rational_column,
    rational_system_status,
    rational_accumulation,
    rational_commit,
    contour_control,
    contour_iteration,
    contour_rank,
    contour_status,
    contour_eigenvalue,
    contour_residual,
    contour_normalization,
};
pub const Algorithm = enum(u64) {
    chebyshev_step = 1,
    chebyshev_series,
    chebyshev_panel,
    lanczos_ritz,
    lanczos_chain,
    operator_apply,
    operator_scaled,
    operator_panel,
    normalized_chebyshev,
    chebyshev_series_set,
    subspace_projection,
    bicgstab,
    rational_filter,
    contour_eigensolver,
    inexact_rational_filter,
    residual_contour_filter,
    inexact_residual_contour_filter,
};
pub const Point = extern struct {
    stage: Stage,
    reserved: u32 = 0,
    index: u64 = 0,
    control: u64 = 0,
};
pub const Synchronization = struct {
    context: *const anyopaque,
    agree_fn: *const fn (*const anyopaque, ?*const execution.context.Operation, Point, ?anyerror) anyerror!void,
};

/// The local result is never masked by a successful callback. Payloads owning
/// resources must have an error cleanup installed before invoking resolve.
pub fn resolve(sync: ?Synchronization, operation: ?*const execution.context.Operation, point: Point, local: anytype) anyerror!@typeInfo(@TypeOf(local)).error_union.payload {
    if (sync) |s| {
        const err: ?anyerror = if (local) |_| null else |failure| failure;
        try s.agree_fn(s.context, operation, point, err);
    }
    return local;
}
pub fn check(sync: ?Synchronization, operation: ?*const execution.context.Operation, point: Point) !void {
    return resolve(sync, operation, point, @as(anyerror!void, {}));
}
pub fn outcome(local: anytype) anyerror!void {
    if (local) |_| return else |failure| return failure;
}
pub fn enter(comptime T: type, sync: ?Synchronization, operation: *const execution.context.Operation, algorithm: Algorithm) !void {
    if (sync == null) return;
    try check(sync, operation, .{ .stage = .entry, .control = @backingInt(algorithm) });
    const kind: u64 = if (T == f32) 1 else if (T == f64) 2 else if (T == scalar.ComplexF32) 3 else if (T == scalar.ComplexF64) 4 else @compileError("unsupported synchronization scalar");
    try check(sync, operation, .{ .stage = .scalar_kind, .control = kind });
}
pub fn realBits(comptime R: type, value: R) u64 {
    return if (R == f32) @as(u32, @bitCast(value)) else if (R == f64) @as(u64, @bitCast(value)) else @compileError("unsupported synchronization real");
}
pub const Parts = struct { real: u64, imaginary: u64 };
pub fn scalarParts(comptime T: type, value: T) Parts {
    return .{
        .real = realBits(scalar.Real(T), if (comptime scalar.isComplex(T)) value.re else value),
        .imaginary = if (comptime scalar.isComplex(T)) realBits(scalar.Real(T), value.im) else 0,
    };
}
pub fn coefficients(comptime T: type, sync: ?Synchronization, operation: *const execution.context.Operation, values: []const T) !void {
    if (sync == null) return;
    try check(sync, operation, .{ .stage = .coefficients_count, .control = @intCast(values.len) });
    for (values, 0..) |value, index| {
        const parts = scalarParts(T, value);
        try check(sync, operation, .{ .stage = .coefficient_real, .index = @intCast(index), .control = parts.real });
        if (comptime scalar.isComplex(T)) try check(sync, operation, .{ .stage = .coefficient_imaginary, .index = @intCast(index), .control = parts.imaginary });
    }
}
pub fn scales(comptime T: type, sync: ?Synchronization, operation: *const execution.context.Operation, alpha: T, beta: T) !void {
    if (sync == null) return;
    const a = scalarParts(T, alpha);
    const b = scalarParts(T, beta);
    try check(sync, operation, .{ .stage = .scale_alpha_real, .control = a.real });
    try check(sync, operation, .{ .stage = .scale_alpha_imaginary, .control = a.imaginary });
    try check(sync, operation, .{ .stage = .scale_beta_real, .control = b.real });
    try check(sync, operation, .{ .stage = .scale_beta_imaginary, .control = b.imaginary });
}
pub fn reductionControls(comptime T: type, sync: ?Synchronization, operation: *const execution.context.Operation, dimension: usize, max_order: usize, has_reduction: bool, hermitian_tolerance: scalar.Real(T)) !void {
    if (sync == null) return;
    try check(sync, operation, .{ .stage = .global_dimension, .control = @intCast(dimension) });
    try check(sync, operation, .{ .stage = .max_order, .control = @intCast(max_order) });
    try check(sync, operation, .{ .stage = .reduction_backend, .control = @intFromBool(has_reduction) });
    try check(sync, operation, .{ .stage = .hermitian_tolerance, .control = realBits(scalar.Real(T), hermitian_tolerance) });
}
pub fn begin(sync: ?Synchronization, owner: *execution.context.Context) !execution.context.Operation {
    var local = owner.begin();
    errdefer if (local) |*operation| operation.end() else |_| {};
    const operation: ?*const execution.context.Operation = if (local) |*value| value else |_| null;
    try resolve(sync, operation, .{ .stage = .admission }, outcome(local));
    return local;
}
pub fn acquire(comptime C: type, comptime T: type, sync: ?Synchronization, operation: *const C.Operation, count: usize, stage: Stage) !C.ContextScratch(T) {
    var local = C.ContextScratch(T).acquire(operation, count);
    errdefer if (local) |*scratch| scratch.deinit() else |_| {};
    return resolve(sync, operation, .{ .stage = stage }, local);
}
