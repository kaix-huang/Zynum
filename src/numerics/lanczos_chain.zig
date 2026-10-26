// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Fixed-depth Hermitian Lanczos coefficients for Haydock continued fractions.
//! No Ritz solve, dimension cap, reorthogonalization, allocation or admission
//! in the numerical leaf. Caller reductions borrow the original operation.
const std = @import("std");
const scalar = @import("../blas/core/shared/scalar.zig");
const storage = @import("storage.zig");
const operators = @import("operator.zig");
const lanczos = @import("lanczos.zig");
const execution = @import("zynum-execution");
const synchronization = @import("synchronization.zig");

pub const Status = enum { depth_limit, breakdown };
pub fn Result(comptime T: type) type {
    return struct { status: Status, order: usize, action_count: usize, residual_norm: scalar.Real(T) };
}
pub fn Request(comptime T: type) type {
    return struct {
        action: operators.Operator(T),
        initial: []const T,
        global_dimension: usize,
        max_order: usize,
        breakdown_tolerance: scalar.Real(T),
        hermitian_tolerance: scalar.Real(T) = 64 * std.math.floatEps(scalar.Real(T)),
        reduction: ?lanczos.Reduction(T) = null,
        alpha: []scalar.Real(T),
        beta: []scalar.Real(T),
    };
}
pub const Lengths = struct { vectors: usize, real: usize };
pub fn Workspace(comptime T: type) type {
    return struct { vectors: []T, real: []scalar.Real(T) };
}
pub fn workspaceLength(comptime T: type, action: operators.Operator(T), max_order: usize) !Lengths {
    if (max_order == 0) return error.InvalidOptions;
    const triples = std.math.mul(usize, action.rows, 3) catch return error.SizeOverflow;
    const vectors = std.math.add(usize, triples, action.workspace_elements) catch return error.SizeOverflow;
    const twice = std.math.mul(usize, max_order, 2) catch return error.SizeOverflow;
    _ = try storage.byteLength(vectors, @sizeOf(T));
    _ = try storage.byteLength(twice - 1, @sizeOf(scalar.Real(T)));
    return .{ .vectors = vectors, .real = twice - 1 };
}
fn finite(comptime T: type, x: T) bool {
    return if (comptime scalar.isComplex(T)) std.math.isFinite(x.re) and std.math.isFinite(x.im) else std.math.isFinite(x);
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
fn preflight(comptime T: type, request: Request(T)) !Lengths {
    const action = request.action;
    if (action.rows != action.columns or request.initial.len != action.rows) return error.DimensionMismatch;
    if (request.global_dimension == 0 or action.rows > request.global_dimension or request.max_order == 0 or
        !std.math.isFinite(request.breakdown_tolerance) or request.breakdown_tolerance < 0 or
        !std.math.isFinite(request.hermitian_tolerance) or request.hermitian_tolerance < 0) return error.InvalidOptions;
    if (request.reduction == null and request.global_dimension != action.rows) return error.MissingGlobalReduction;
    const used = try workspaceLength(T, action, request.max_order);
    if (request.alpha.len < request.max_order or request.beta.len < request.max_order - 1) return error.BufferTooSmall;
    // Validate full declared output spans, including tails, before touching any
    // input or invoking caller code. A caller cannot hide an alias in a tail.
    try noOverlap(request.alpha, request.beta);
    try noOverlap(request.alpha, request.initial);
    try noOverlap(request.beta, request.initial);
    for (request.initial) |x| if (!finite(T, x)) return error.NonFiniteInput;
    return used;
}
fn inner(comptime T: type, operation: *const execution.context.Operation, request: Request(T), x: []const T, y: []const T) !T {
    const local = innerLocal(T, operation, request, x, y);
    const bits: synchronization.Parts = if (local) |value| synchronization.scalarParts(T, value) else |_| .{ .real = 0, .imaginary = 0 };
    const result = try synchronization.resolve(request.action.synchronization, operation, .{ .stage = .projection_real, .control = bits.real }, local);
    try synchronization.check(request.action.synchronization, operation, .{ .stage = .projection_imaginary, .control = bits.imaginary });
    return result;
}
fn innerLocal(comptime T: type, operation: *const execution.context.Operation, request: Request(T), x: []const T, y: []const T) !T {
    const out = if (request.reduction) |r| try r.inner_fn(r.context, operation, x, y) else try lanczos.localInner(T, x, y);
    if (!finite(T, out)) return error.NonFiniteReduction;
    return out;
}
fn norm(comptime T: type, operation: *const execution.context.Operation, request: Request(T), x: []const T, stage: synchronization.Stage, index: u64) !scalar.Real(T) {
    const local = normLocal(T, operation, request, x);
    const bits = if (local) |value| synchronization.realBits(scalar.Real(T), value) else |_| 0;
    return synchronization.resolve(request.action.synchronization, operation, .{ .stage = stage, .index = index, .control = bits }, local);
}
fn normLocal(comptime T: type, operation: *const execution.context.Operation, request: Request(T), x: []const T) !scalar.Real(T) {
    const out = if (request.reduction) |r| try r.norm_fn(r.context, operation, x) else try lanczos.localNorm(T, x);
    if (!std.math.isFinite(out) or out < 0) return error.NonFiniteReduction;
    return out;
}
fn subtractReal(comptime T: type, x: T, factor: scalar.Real(T), y: T) !T {
    const W = wide(T);
    const out: T = if (comptime scalar.isComplex(T)) .{
        .re = @floatCast(@as(W, x.re) - @as(W, factor) * y.re),
        .im = @floatCast(@as(W, x.im) - @as(W, factor) * y.im),
    } else @floatCast(@as(W, x) - @as(W, factor) * y);
    if (!finite(T, out)) return error.NonFiniteResult;
    return out;
}
fn divideReal(comptime T: type, x: T, divisor: scalar.Real(T)) !T {
    const W = wide(T);
    const out: T = if (comptime scalar.isComplex(T)) .{
        .re = @floatCast(@as(W, x.re) / divisor),
        .im = @floatCast(@as(W, x.im) / divisor),
    } else @floatCast(@as(W, x) / divisor);
    if (!finite(T, out)) return error.NonFiniteResult;
    return out;
}
fn divideValues(comptime T: type, source: []const T, destination: []T, length: scalar.Real(T)) !void {
    for (destination, source) |*q, x| q.* = try divideReal(T, x, length);
}
fn subtractValues(comptime T: type, values: []T, factor: scalar.Real(T), source: []const T) !void {
    for (values, source) |*x, y| x.* = try subtractReal(T, x.*, factor, y);
}
fn normalize(comptime T: type, operation: *const execution.context.Operation, request: Request(T), source: []const T, destination: []T, length: scalar.Real(T), index: u64) !void {
    try synchronization.resolve(request.action.synchronization, operation, .{ .stage = .local_arithmetic, .index = index }, divideValues(T, source, destination, length));
    // Correct rounding of a subnormal scalar norm has coarse absolute spacing.
    // Remeasure the scaled vector through the SAME borrowed global/local metric,
    // on every shard. Preserve the original scalar coefficient in the chain.
    if (length < std.math.floatMin(scalar.Real(T))) {
        const unit = try norm(T, operation, request, destination, .unit_norm, index);
        if (unit == 0) return error.NonFiniteReduction;
        try synchronization.resolve(request.action.synchronization, operation, .{ .stage = .local_arithmetic, .index = index }, divideValues(T, destination, destination, unit));
    }
}
fn checkedWorkspace(comptime T: type, request: Request(T), workspace: Workspace(T)) !Lengths {
    const used = try preflight(T, request);
    if (workspace.vectors.len < used.vectors or workspace.real.len < used.real) return error.WorkspaceTooSmall;
    const work = workspace.vectors[0..used.vectors];
    const real_work = workspace.real[0..used.real];
    try noOverlap(request.initial, work);
    try noOverlap(request.initial, real_work);
    try noOverlap(work, real_work);
    try noOverlap(request.alpha, work);
    try noOverlap(request.alpha, real_work);
    try noOverlap(request.beta, work);
    try noOverlap(request.beta, real_work);
    return used;
}
fn removeProjection(comptime T: type, request: Request(T), residual: []T, current: []const T, projection: T) !scalar.Real(T) {
    const alpha = if (comptime scalar.isComplex(T)) projection.re else projection;
    const imaginary: scalar.Real(T) = if (comptime scalar.isComplex(T)) projection.im else 0;
    const W = wide(T);
    if (@as(W, @abs(imaginary)) > @as(W, request.hermitian_tolerance) * @max(@as(W, 1), @abs(@as(W, alpha)))) return error.NonHermitianAction;
    try subtractValues(T, residual, alpha, current);
    return alpha;
}
/// max_order counts actions; it may exceed the global dimension. Exact zero or
/// residual < breakdown_tolerance stops the recurrence (strict comparison).
/// Success commits only alpha[0..order] and beta[0..order-1]. Every failure leaves
/// the entire declared alpha/beta spans unchanged; scratch and callbacks may
/// have changed. Input and action data remain borrowed through this return.
pub fn runAdmitted(comptime T: type, operation: *const execution.context.Operation, request: Request(T), workspace: Workspace(T)) !Result(T) {
    _ = operation.borrowedContext();
    const sync = request.action.synchronization;
    try synchronization.enter(T, sync, operation, .lanczos_chain);
    const used = try synchronization.resolve(sync, operation, .{ .stage = .preflight }, checkedWorkspace(T, request, workspace));
    try synchronization.reductionControls(T, sync, operation, request.global_dimension, request.max_order, request.reduction != null, request.hermitian_tolerance);
    try synchronization.check(sync, operation, .{ .stage = .breakdown_tolerance, .control = synchronization.realBits(scalar.Real(T), request.breakdown_tolerance) });
    const work = workspace.vectors[0..used.vectors];
    const real_work = workspace.real[0..used.real];
    const n = request.action.rows;
    const m = request.max_order;
    const callback_work = work[0..request.action.workspace_elements];
    const vectors = work[request.action.workspace_elements..];
    const previous = vectors[0..n];
    const current = vectors[n..][0..n];
    const residual = vectors[2 * n ..][0..n];
    const diagonal = real_work[0..m];
    const off = real_work[m..][0 .. m - 1];
    const initial_norm = try norm(T, operation, request, request.initial, .initial_norm, 0);
    if (initial_norm == 0) return error.InvalidInitialVector;
    @memset(previous, scalar.zero(T));
    try normalize(T, operation, request, request.initial, current, initial_norm, 0);
    var beta: scalar.Real(T) = 0;
    for (0..m) |step| {
        try request.action.applyAdmitted(operation, current, residual, callback_work);
        try synchronization.resolve(sync, operation, .{ .stage = .local_arithmetic, .index = @intCast(step) }, subtractValues(T, residual, beta, previous));
        const projection = try inner(T, operation, request, current, residual);
        const alpha = try synchronization.resolve(sync, operation, .{ .stage = .local_arithmetic, .index = @intCast(step) }, removeProjection(T, request, residual, current, projection));
        diagonal[step] = alpha;
        if (step > 0) off[step - 1] = beta;
        beta = try norm(T, operation, request, residual, .residual_norm, @intCast(step));
        const order = step + 1;
        const breakdown = beta == 0 or beta < request.breakdown_tolerance;
        const stopping = breakdown or order == m;
        const final_status: Status = if (breakdown) .breakdown else .depth_limit;
        try synchronization.check(sync, operation, .{ .stage = .stopping_decision, .index = @intCast(order), .control = if (stopping) @as(u64, @backingInt(final_status)) + 1 else 0 });
        if (stopping) {
            try synchronization.check(sync, operation, .{ .stage = .commit, .index = @intCast(order) });
            @memcpy(request.alpha[0..order], diagonal[0..order]);
            @memcpy(request.beta[0 .. order - 1], off[0 .. order - 1]);
            return .{ .status = final_status, .order = order, .action_count = order, .residual_norm = beta };
        }
        @memcpy(previous, current);
        try normalize(T, operation, request, residual, current, beta, @intCast(order));
    }
    unreachable;
}
pub fn Managed(comptime Execution: type) type {
    return struct {
        const context = Execution.context;
        pub fn runBorrowedAdmitted(comptime T: type, operation: *const context.Operation, request: Request(T), workspace: Workspace(T)) !Result(T) {
            return runAdmitted(T, operation, request, workspace);
        }
        pub fn runScratchAdmitted(comptime T: type, operation: *const context.Operation, request: Request(T)) !Result(T) {
            _ = operation.borrowedContext();
            try synchronization.enter(T, request.action.synchronization, operation, .lanczos_chain);
            const used = try synchronization.resolve(request.action.synchronization, operation, .{ .stage = .workspace }, preflight(T, request));
            var vectors = try synchronization.acquire(context, T, request.action.synchronization, operation, used.vectors, .vector_allocation);
            defer vectors.deinit();
            var real = try synchronization.acquire(context, scalar.Real(T), request.action.synchronization, operation, used.real, .real_allocation);
            defer real.deinit();
            return runAdmitted(T, operation, request, .{ .vectors = vectors.lease.values, .real = real.lease.values });
        }
        pub fn runBorrowed(comptime T: type, owner: *context.Context, request: Request(T), workspace: Workspace(T)) !Result(T) {
            var operation = try synchronization.begin(request.action.synchronization, owner);
            defer operation.end();
            return runBorrowedAdmitted(T, &operation, request, workspace);
        }
        pub fn runScratch(comptime T: type, owner: *context.Context, request: Request(T)) !Result(T) {
            var operation = try synchronization.begin(request.action.synchronization, owner);
            defer operation.end();
            return runScratchAdmitted(T, &operation, request);
        }
    };
}
