// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Three-term Hermitian Lanczos with native STEV Ritz values. No implicit MPI,
//! initial-probe policy, spectrum padding, reorthogonalization or new resource
//! owner. Reductions borrow caller code/state through the synchronous return.
const std = @import("std");
const scalar = @import("../blas/core/shared/scalar.zig");
const storage = @import("storage.zig");
const operators = @import("operator.zig");
const execution = @import("zynum-execution");
const stev = @import("../lapack/core/tridiagonal_eigen.zig");
const synchronization = @import("synchronization.zig");

pub const Status = enum { converged, exhausted_dimension, early_breakdown, iteration_limit };
pub fn Result(comptime T: type) type {
    return struct { status: Status, order: usize, action_count: usize, minimum: scalar.Real(T), maximum: scalar.Real(T), has_previous_ritz: bool, min_change: scalar.Real(T), max_change: scalar.Real(T), residual_norm: scalar.Real(T) };
}
pub const Reduction = @import("reductions.zig").Reduction;
pub fn Request(comptime T: type) type {
    return struct {
        action: operators.Operator(T),
        initial: []const T,
        global_dimension: usize,
        max_order: usize,
        ritz_tolerance: scalar.Real(T),
        hermitian_tolerance: scalar.Real(T) = 64 * std.math.floatEps(scalar.Real(T)),
        reduction: ?Reduction(T) = null,
        ritz_max_iterations: ?usize = null,
    };
}
pub const Lengths = struct { vectors: usize, real: usize };
pub fn Workspace(comptime T: type) type {
    return struct { vectors: []T, real: []scalar.Real(T) };
}
pub fn workspaceLength(comptime T: type, action: operators.Operator(T), max_order: usize) !Lengths {
    if (max_order == 0) return error.InvalidOptions;
    const n = std.math.mul(usize, action.rows, 3) catch return error.SizeOverflow;
    const vectors = std.math.add(usize, n, action.workspace_elements) catch return error.SizeOverflow;
    const real = std.math.mul(usize, max_order, 4) catch return error.SizeOverflow;
    _ = try storage.byteLength(vectors, @sizeOf(T));
    _ = try storage.byteLength(real - 2, @sizeOf(scalar.Real(T)));
    return .{ .vectors = vectors, .real = real - 2 };
}
fn finite(comptime T: type, x: T) bool {
    return if (comptime scalar.isComplex(T)) std.math.isFinite(x.re) and std.math.isFinite(x.im) else std.math.isFinite(x);
}
fn realPart(comptime T: type, x: T) scalar.Real(T) {
    return if (comptime scalar.isComplex(T)) x.re else x;
}
fn imaginary(comptime T: type, x: T) scalar.Real(T) {
    return if (comptime scalar.isComplex(T)) x.im else 0;
}
fn wide(comptime T: type) type {
    return if (scalar.Real(T) == f32) f64 else f128;
}
fn range(values: anytype) !storage.ByteRange {
    return storage.byteRange(@intFromPtr(values.ptr), try storage.byteLength(values.len, @sizeOf(@TypeOf(values[0]))));
}
fn noOverlap(first: anytype, second: anytype) !void {
    if (storage.overlaps(try range(first), try range(second))) return error.AliasingNotAllowed;
}
fn checkFinite(comptime T: type, values: []const T) !void {
    for (values) |x| if (!finite(T, x)) return error.NonFiniteInput;
}
// Preserve the established composition seam through the common implementation.
pub const localInner = @import("reductions.zig").localInner;
pub const localNorm = @import("reductions.zig").localNorm;
fn inner(comptime T: type, operation: *const execution.context.Operation, request: Request(T), x: []const T, y: []const T) !T {
    const local = innerLocal(T, operation, request, x, y);
    const bits: synchronization.Parts = if (local) |value| synchronization.scalarParts(T, value) else |_| .{ .real = 0, .imaginary = 0 };
    const result = try synchronization.resolve(request.action.synchronization, operation, .{ .stage = .projection_real, .control = bits.real }, local);
    try synchronization.check(request.action.synchronization, operation, .{ .stage = .projection_imaginary, .control = bits.imaginary });
    return result;
}
fn innerLocal(comptime T: type, operation: *const execution.context.Operation, request: Request(T), x: []const T, y: []const T) !T {
    return @import("reductions.zig").inner(T, operation, request.reduction, x, y);
}
fn norm(comptime T: type, operation: *const execution.context.Operation, request: Request(T), x: []const T, stage: synchronization.Stage, index: u64) !scalar.Real(T) {
    const local = normLocal(T, operation, request, x);
    const bits = if (local) |value| synchronization.realBits(scalar.Real(T), value) else |_| 0;
    return synchronization.resolve(request.action.synchronization, operation, .{ .stage = stage, .index = index, .control = bits }, local);
}
fn normLocal(comptime T: type, operation: *const execution.context.Operation, request: Request(T), x: []const T) !scalar.Real(T) {
    return @import("reductions.zig").norm(T, operation, request.reduction, x);
}
fn subtractReal(comptime T: type, x: T, factor: scalar.Real(T), y: T) !T {
    const W = wide(T);
    const out: T = if (comptime scalar.isComplex(T)) .{ .re = @floatCast(@as(W, x.re) - @as(W, factor) * y.re), .im = @floatCast(@as(W, x.im) - @as(W, factor) * y.im) } else @floatCast(@as(W, x) - @as(W, factor) * y);
    if (!finite(T, out)) return error.NonFiniteResult;
    return out;
}
fn divideReal(comptime T: type, x: T, divisor: scalar.Real(T)) !T {
    const W = wide(T);
    const out: T = if (comptime scalar.isComplex(T)) .{ .re = @floatCast(@as(W, x.re) / divisor), .im = @floatCast(@as(W, x.im) / divisor) } else @floatCast(@as(W, x) / divisor);
    if (!finite(T, out)) return error.NonFiniteResult;
    return out;
}
fn divideValues(comptime T: type, source: []const T, destination: []T, divisor: scalar.Real(T)) !void {
    for (destination, source) |*q, x| q.* = try divideReal(T, x, divisor);
}
fn subtractValues(comptime T: type, values: []T, factor: scalar.Real(T), source: []const T) !void {
    for (values, source) |*x, y| x.* = try subtractReal(T, x.*, factor, y);
}
fn preflight(comptime T: type, request: Request(T), workspace: Workspace(T)) !Lengths {
    const action = request.action;
    const n = action.rows;
    const m = request.max_order;
    if (action.rows != action.columns or request.initial.len != n) return error.DimensionMismatch;
    if (request.global_dimension == 0 or n > request.global_dimension or m == 0 or !std.math.isFinite(request.ritz_tolerance) or request.ritz_tolerance < 0 or !std.math.isFinite(request.hermitian_tolerance) or request.hermitian_tolerance < 0) return error.InvalidOptions;
    if (request.reduction == null and request.global_dimension != n) return error.MissingGlobalReduction;
    const used = try workspaceLength(T, action, m);
    if (workspace.vectors.len < used.vectors or workspace.real.len < used.real) return error.WorkspaceTooSmall;
    const work = workspace.vectors[0..used.vectors];
    const real_work = workspace.real[0..used.real];
    try noOverlap(request.initial, work);
    try noOverlap(request.initial, real_work);
    try noOverlap(work, real_work);
    try checkFinite(T, request.initial);
    return used;
}
fn ritzUpdate(comptime T: type, request: Request(T), residual: []T, current: []const T, projection: T, diagonal: []scalar.Real(T), off: []scalar.Real(T), values: []scalar.Real(T), ql_off: []scalar.Real(T), step: usize, beta: scalar.Real(T), old_min: scalar.Real(T), old_max: scalar.Real(T)) !struct { minimum: scalar.Real(T), maximum: scalar.Real(T), min_change: scalar.Real(T), max_change: scalar.Real(T) } {
    const R = scalar.Real(T);
    const alpha = realPart(T, projection);
    if (@abs(imaginary(T, projection)) > request.hermitian_tolerance * @max(@as(R, 1), @abs(alpha))) return error.NonHermitianAction;
    try subtractValues(T, residual, alpha, current);
    diagonal[step] = alpha;
    if (step > 0) off[step - 1] = beta;
    const order = step + 1;
    @memcpy(values[0..order], diagonal[0..order]);
    @memcpy(ql_off[0 .. order - 1], off[0 .. order - 1]);
    const ritz = stev.solve(R, values[0..order], ql_off[0 .. order - 1], null, .{ .max_iterations = request.ritz_max_iterations });
    if (ritz.status != .converged) return error.RitzSolverFailure;
    const minimum = values[0];
    const maximum = values[order - 1];
    if (!std.math.isFinite(minimum) or !std.math.isFinite(maximum)) return error.NonFiniteResult;
    return .{ .minimum = minimum, .maximum = maximum, .min_change = if (step > 0) @abs(minimum - old_min) else 0, .max_change = if (step > 0) @abs(maximum - old_max) else 0 };
}
/// global_dimension describes the actual operator domain, not local shard size.
/// Distributed callers must provide both global reductions even on empty ranks.
/// max_order counts actions, so ABPLaS's inclusive step budget maps to budget+1.
/// Early breakdown is reported separately: an invariant subspace does not prove
/// the full spectrum. Returned converged Ritz ends are estimates, not rigorous
/// spectrum bounds. Initial vector and opaque numerical action data stay fixed;
/// scratch changes on error. No result is published when a callback fails.
pub fn runAdmitted(comptime T: type, operation: *const execution.context.Operation, request: Request(T), workspace: Workspace(T)) !Result(T) {
    _ = operation.borrowedContext();
    const R = scalar.Real(T);
    const action = request.action;
    const n = action.rows;
    const m = request.max_order;
    const sync = action.synchronization;
    try synchronization.enter(T, sync, operation, .lanczos_ritz);
    const used = try synchronization.resolve(sync, operation, .{ .stage = .preflight }, preflight(T, request, workspace));
    try synchronization.reductionControls(T, sync, operation, request.global_dimension, m, request.reduction != null, request.hermitian_tolerance);
    try synchronization.check(sync, operation, .{ .stage = .ritz_tolerance, .control = synchronization.realBits(R, request.ritz_tolerance) });
    try synchronization.check(sync, operation, .{ .stage = .ritz_budget_present, .control = @intFromBool(request.ritz_max_iterations != null) });
    if (request.ritz_max_iterations) |budget| try synchronization.check(sync, operation, .{ .stage = .ritz_budget, .control = @intCast(budget) });
    const work = workspace.vectors[0..used.vectors];
    const real_work = workspace.real[0..used.real];
    const callback_work = work[0..action.workspace_elements];
    const vectors = work[action.workspace_elements..];
    const previous = vectors[0..n];
    const current = vectors[n..][0..n];
    const residual = vectors[2 * n ..][0..n];
    const diagonal = real_work[0..m];
    const off = real_work[m..][0 .. m - 1];
    const values = real_work[2 * m - 1 ..][0..m];
    const ql_off = real_work[3 * m - 1 ..][0 .. m - 1];
    const initial_norm = try norm(T, operation, request, request.initial, .initial_norm, 0);
    if (initial_norm == 0) return error.InvalidInitialVector;
    @memset(previous, scalar.zero(T));
    try synchronization.resolve(sync, operation, .{ .stage = .local_arithmetic, .index = std.math.maxInt(u64) }, divideValues(T, request.initial, current, initial_norm));
    // A subnormal norm has coarse absolute spacing: its correctly rounded
    // scalar value can still leave x/norm far from unit length. Remeasure the
    // safely scaled vector through the same local or borrowed global metric.
    // Every distributed shard follows this branch using the same global norm.
    if (initial_norm < std.math.floatMin(R)) {
        const unit_norm = try norm(T, operation, request, current, .unit_norm, 0);
        if (unit_norm == 0) return error.NonFiniteReduction;
        try synchronization.resolve(sync, operation, .{ .stage = .local_arithmetic, .index = std.math.maxInt(u64) - 1 }, divideValues(T, current, current, unit_norm));
    }
    var beta: R = 0;
    var old_min: R = 0;
    var old_max: R = 0;
    for (0..m) |step| {
        try action.applyAdmitted(operation, current, residual, callback_work);
        try synchronization.resolve(sync, operation, .{ .stage = .local_arithmetic, .index = @intCast(step) }, subtractValues(T, residual, beta, previous));
        const projection = try inner(T, operation, request, current, residual);
        const order = step + 1;
        const ritz = try synchronization.resolve(sync, operation, .{ .stage = .local_arithmetic, .index = @intCast(order) }, ritzUpdate(T, request, residual, current, projection, diagonal, off, values, ql_off, step, beta, old_min, old_max));
        const minimum = ritz.minimum;
        const maximum = ritz.maximum;
        const min_change = ritz.min_change;
        const max_change = ritz.max_change;
        inline for (.{ .{ synchronization.Stage.minimum_ritz, minimum }, .{ synchronization.Stage.maximum_ritz, maximum }, .{ synchronization.Stage.minimum_change, min_change }, .{ synchronization.Stage.maximum_change, max_change } }) |field| {
            try synchronization.check(sync, operation, .{ .stage = field[0], .index = @intCast(order), .control = synchronization.realBits(R, field[1]) });
        }
        beta = try norm(T, operation, request, residual, .residual_norm, @intCast(order));
        const threshold = 64 * std.math.floatEps(R) * @max(@as(R, 1), @max(@abs(minimum), @abs(maximum)));
        const breakdown = beta <= threshold;
        const converged = step > 0 and min_change <= request.ritz_tolerance and max_change <= request.ritz_tolerance;
        const stopping = breakdown or converged or order == m;
        const final_status: Status = if (breakdown) (if (order == request.global_dimension) .exhausted_dimension else .early_breakdown) else if (converged) .converged else .iteration_limit;
        try synchronization.check(sync, operation, .{ .stage = .stopping_decision, .index = @intCast(order), .control = if (stopping) @as(u64, @backingInt(final_status)) + 1 else 0 });
        if (stopping) {
            try synchronization.check(sync, operation, .{ .stage = .commit, .index = @intCast(order) });
            return .{ .status = final_status, .order = order, .action_count = order, .minimum = minimum, .maximum = maximum, .has_previous_ritz = step > 0, .min_change = min_change, .max_change = max_change, .residual_norm = beta };
        }
        old_min = minimum;
        old_max = maximum;
        @memcpy(previous, current);
        try synchronization.resolve(sync, operation, .{ .stage = .local_arithmetic, .index = @intCast(order) }, divideValues(T, residual, current, beta));
    }
    unreachable;
}
pub fn Managed(comptime Execution: type) type {
    return struct {
        const context = Execution.context;
        pub fn runBorrowedAdmitted(comptime T: type, operation: *const context.Operation, request: Request(T), work: Workspace(T)) !Result(T) {
            return runAdmitted(T, operation, request, work);
        }
        pub fn runScratchAdmitted(comptime T: type, operation: *const context.Operation, request: Request(T)) !Result(T) {
            _ = operation.borrowedContext();
            try synchronization.enter(T, request.action.synchronization, operation, .lanczos_ritz);
            const used = try synchronization.resolve(request.action.synchronization, operation, .{ .stage = .workspace }, workspaceLength(T, request.action, request.max_order));
            var vectors = try synchronization.acquire(context, T, request.action.synchronization, operation, used.vectors, .vector_allocation);
            defer vectors.deinit();
            var real = try synchronization.acquire(context, scalar.Real(T), request.action.synchronization, operation, used.real, .real_allocation);
            defer real.deinit();
            return runAdmitted(T, operation, request, .{ .vectors = vectors.lease.values, .real = real.lease.values });
        }
        pub fn runBorrowed(comptime T: type, owner: *context.Context, request: Request(T), work: Workspace(T)) !Result(T) {
            var operation = try synchronization.begin(request.action.synchronization, owner);
            defer operation.end();
            return runBorrowedAdmitted(T, &operation, request, work);
        }
        pub fn runScratch(comptime T: type, owner: *context.Context, request: Request(T)) !Result(T) {
            var operation = try synchronization.begin(request.action.synchronization, owner);
            defer operation.end();
            return runScratchAdmitted(T, &operation, request);
        }
    };
}
