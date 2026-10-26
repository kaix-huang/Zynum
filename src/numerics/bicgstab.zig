// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Native operator BiCGStab with optional stationary right preconditioning.
//! Caller communication is borrowed through common scalar reductions and phase
//! agreement. Recurrence coefficients remain widened; normalized correction
//! residuals avoid squaring the physical RHS scale. All controlled stops verify
//! b-A*x. False recursive convergence restarts from that residual within the
//! original iteration budget. Errors preserve output; a controlled stop commits
//! its finite candidate. No allocator, thread, handle or communicator owner.
const std = @import("std");
const S = @import("../blas/core/shared/scalar.zig");
const O = @import("operator.zig");
const R = @import("reductions.zig");
const storage = @import("storage.zig");
const arithmetic = @import("recurrence_arithmetic.zig");
const execution = @import("zynum-execution");
const P = @import("synchronization.zig");

pub const Status = enum { converged, iteration_limit, rho_breakdown, alpha_breakdown, omega_breakdown };
pub fn Result(comptime T: type) type {
    return struct { status: Status, iterations: usize, action_count: usize, preconditioner_count: usize, residual_restarts: usize, residual_norm: S.Real(T) };
}
pub const Reduction = R.Reduction;
pub fn Request(comptime T: type) type {
    return struct {
        action: O.Operator(T),
        action_scale: T = S.one(T),
        diagonal_shift: T = S.zero(T),
        preconditioner: ?O.Operator(T) = null,
        reduction: ?Reduction(T) = null,
        rhs: []const T,
        initial: ?[]const T = null,
        output: []T,
        global_dimension: usize,
        max_iterations: usize,
        relative_tolerance: S.Real(T),
        absolute_tolerance: S.Real(T) = 0,
    };
}
pub fn workspaceLength(comptime T: type, action: O.Operator(T), preconditioner: ?O.Operator(T)) !usize {
    const extra = if (preconditioner) |p| @max(action.workspace_elements, p.workspace_elements) else action.workspace_elements;
    const vectors = std.math.mul(usize, action.rows, 9) catch return error.SizeOverflow;
    const count = std.math.add(usize, vectors, extra) catch return error.SizeOverflow;
    _ = try storage.byteLength(count, @sizeOf(T));
    return count;
}
fn range(values: anytype) !storage.ByteRange {
    return storage.byteRange(@intFromPtr(values.ptr), try storage.byteLength(values.len, @sizeOf(@typeInfo(@TypeOf(values)).pointer.child)));
}
fn separate(a: anytype, b: anytype) !void {
    if (storage.overlaps(try range(a), try range(b))) return error.AliasingNotAllowed;
}
fn checkFinite(comptime T: type, values: []const T, input: bool) !void {
    for (values) |v| if (!R.finite(T, v)) return if (input) error.NonFiniteInput else error.NonFiniteResult;
}
fn options(comptime T: type, q: Request(T)) !void {
    const n = q.action.rows;
    if (!R.finite(T, q.action_scale) or !R.finite(T, q.diagonal_shift)) return error.NonFiniteInput;
    if (q.action.columns != n or q.rhs.len != n or q.output.len < n or n > q.global_dimension) return error.DimensionMismatch;
    if (q.initial) |initial| if (initial.len != n) return error.DimensionMismatch;
    if (q.reduction == null and q.global_dimension != n) return error.MissingGlobalReduction;
    if (!std.math.isFinite(q.relative_tolerance) or q.relative_tolerance < 0 or !std.math.isFinite(q.absolute_tolerance) or q.absolute_tolerance < 0) return error.InvalidOptions;
    if (q.preconditioner) |p| {
        if (p.rows != n or p.columns != n) return error.DimensionMismatch;
        if (p.synchronization) |sync| {
            const parent = q.action.synchronization orelse return error.InvalidOptions;
            if (sync.context != parent.context or sync.agree_fn != parent.agree_fn) return error.InvalidOptions;
        }
    }
    try checkFinite(T, q.rhs, true);
    if (q.initial) |initial| try checkFinite(T, initial, true);
    try separate(q.rhs, q.output[0..n]);
}
fn preflight(comptime T: type, q: Request(T), workspace: []T) ![]T {
    try options(T, q);
    const count = try workspaceLength(T, q.action, q.preconditioner);
    if (workspace.len < count) return error.WorkspaceTooSmall;
    const used = workspace[0..count];
    try separate(used, q.rhs);
    try separate(used, q.output[0..q.action.rows]);
    if (q.initial) |initial| try separate(used, initial);
    return used;
}
fn Wide(comptime T: type) type {
    return arithmetic.Wide(T);
}
fn Coefficient(comptime T: type) type {
    return if (S.isComplex(T)) std.math.Complex(Wide(T)) else Wide(T);
}
fn wide(comptime T: type, x: T) Coefficient(T) {
    return if (comptime S.isComplex(T)) .{ .re = x.re, .im = x.im } else x;
}
fn real(comptime T: type, x: Wide(T)) Coefficient(T) {
    return if (comptime S.isComplex(T)) .{ .re = x, .im = 0 } else x;
}
fn add(comptime T: type, a: Coefficient(T), b: Coefficient(T)) Coefficient(T) {
    return if (comptime S.isComplex(T)) .{ .re = a.re + b.re, .im = a.im + b.im } else a + b;
}
fn negate(comptime T: type, a: Coefficient(T)) Coefficient(T) {
    return if (comptime S.isComplex(T)) .{ .re = -a.re, .im = -a.im } else -a;
}
fn mul(comptime T: type, a: Coefficient(T), b: Coefficient(T)) Coefficient(T) {
    return if (comptime S.isComplex(T)) .{ .re = a.re * b.re - a.im * b.im, .im = a.re * b.im + a.im * b.re } else a * b;
}
fn zero(comptime T: type, a: Coefficient(T)) bool {
    return if (comptime S.isComplex(T)) a.re == 0 and a.im == 0 else a == 0;
}
fn divide(comptime T: type, a: Coefficient(T), b: Coefficient(T)) !Coefficient(T) {
    if (zero(T, b)) return error.InvalidDivisor;
    const value: Coefficient(T) = if (comptime S.isComplex(T)) blk: {
        if (@abs(b.re) >= @abs(b.im)) {
            const ratio = b.im / b.re;
            const denominator = b.re + b.im * ratio;
            break :blk .{ .re = (a.re + a.im * ratio) / denominator, .im = (a.im - a.re * ratio) / denominator };
        } else {
            const ratio = b.re / b.im;
            const denominator = b.im + b.re * ratio;
            break :blk .{ .re = (a.re * ratio + a.im) / denominator, .im = (a.im * ratio - a.re) / denominator };
        }
    } else a / b;
    if (comptime S.isComplex(T)) {
        if (!std.math.isFinite(value.re) or !std.math.isFinite(value.im)) return error.NonFiniteResult;
    } else if (!std.math.isFinite(value)) return error.NonFiniteResult;
    return value;
}
fn narrow(comptime T: type, value: Coefficient(T)) !T {
    return arithmetic.produced(T, if (comptime S.isComplex(T)) .{ .re = @floatCast(value.re), .im = @floatCast(value.im) } else @floatCast(value));
}
fn directInnerRange(comptime T: type, nx: S.Real(T), ny: S.Real(T)) bool {
    const product = @as(Wide(T), nx) * ny;
    return product >= std.math.floatMin(S.Real(T)) and product <= std.math.floatMax(S.Real(T));
}
fn normalized(comptime T: type, x: []const T, out: []T, norm: S.Real(T)) !void {
    for (x, out) |value, *target| target.* = try narrow(T, try divide(T, wide(T, value), real(T, norm)));
}
fn State(comptime T: type) type {
    return struct {
        const Self = @This();
        op: *const execution.context.Operation,
        q: Request(T),
        x: []T,
        r: []T,
        shadow: []T,
        p: []T,
        v: []T,
        s: []T,
        t: []T,
        phat: []T,
        shat: []T,
        work: []T,
        iterations: usize = 0,
        actions: usize = 0,
        preconditions: usize = 0,
        restarts: usize = 0,
        sequence: u64 = 0,
        scale: S.Real(T) = 1,
        target: Wide(T) = 0,
        fn point(self: *Self, stage: P.Stage, control: u64) !P.Point {
            self.sequence = std.math.add(u64, self.sequence, 1) catch return error.SizeOverflow;
            return .{ .stage = stage, .index = self.sequence, .control = control };
        }
        fn sync(self: *Self) ?P.Synchronization {
            return self.q.action.synchronization;
        }
        fn local(self: *Self, value: anytype) !@typeInfo(@TypeOf(value)).error_union.payload {
            return P.resolve(self.sync(), self.op, try self.point(.local_arithmetic, 0), value);
        }
        fn norm(self: *Self, x: []const T) !S.Real(T) {
            const value = R.norm(T, self.op, self.q.reduction, x);
            const bits = if (value) |v| P.realBits(S.Real(T), v) else |_| 0;
            return P.resolve(self.sync(), self.op, try self.point(.residual_norm, bits), value);
        }
        fn innerUnits(self: *Self, x: []const T, y: []const T) !T {
            const value = R.inner(T, self.op, self.q.reduction, x, y);
            const bits = if (value) |v| P.scalarParts(T, v) else |_| P.Parts{ .real = 0, .imaginary = 0 };
            const result = try P.resolve(self.sync(), self.op, try self.point(.projection_real, bits.real), value);
            try P.check(self.sync(), self.op, try self.point(.projection_imaginary, bits.imaginary));
            return result;
        }
        fn inner(self: *Self, x: []const T, y: []const T, sx: []T, sy: []T) !Coefficient(T) {
            const nx = try self.norm(x);
            const ny = try self.norm(y);
            if (nx == 0 or ny == 0) return real(T, 0);
            // Preserve ordinary inner-product rounding and exact orthogonality.
            // Normalize only when the product's range requires it.
            if (directInnerRange(T, nx, ny)) return wide(T, try self.innerUnits(x, y));
            try self.local(normalized(T, x, sx, nx));
            try self.local(normalized(T, y, sy, ny));
            return mul(T, wide(T, try self.innerUnits(sx, sy)), real(T, @as(Wide(T), nx) * ny));
        }
        fn affine(self: *Self, x: []const T, y: []T, ignore_action: bool) !void {
            for (x, y) |input, *output| output.* = if (ignore_action)
                try arithmetic.linear(T, self.q.diagonal_shift, input, S.zero(T), S.zero(T))
            else
                try arithmetic.linear(T, self.q.action_scale, output.*, self.q.diagonal_shift, input);
        }
        fn apply(self: *Self, x: []const T, y: []T, is_preconditioner: bool) !void {
            if (!is_preconditioner and S.isZero(T, self.q.action_scale)) return self.local(self.affine(x, y, true));
            const action = if (is_preconditioner) self.q.preconditioner.? else self.q.action;
            const result = action.apply_fn(action.context, self.op, x, y, self.work[0..action.workspace_elements]);
            try P.resolve(self.sync(), self.op, try self.point(if (is_preconditioner) .preconditioner_complete else .action_complete, 0), result);
            if (is_preconditioner) self.preconditions = std.math.add(usize, self.preconditions, 1) catch return error.SizeOverflow else self.actions = std.math.add(usize, self.actions, 1) catch return error.SizeOverflow;
            try self.local(checkFinite(T, y, false));
            if (!is_preconditioner and (!S.isOne(T, self.q.action_scale) or !S.isZero(T, self.q.diagonal_shift))) try self.local(self.affine(x, y, false));
        }
        fn precondition(self: *Self, x: []const T, y: []T) !void {
            if (self.q.preconditioner != null) try self.apply(x, y, true) else @memcpy(y, x);
        }
        fn residual(self: *Self) !S.Real(T) {
            try self.apply(self.x, self.t, false);
            try self.local(self.residualLocal());
            return self.norm(self.r);
        }
        fn residualLocal(self: *Self) !void {
            for (self.q.rhs, self.t, self.r) |b, ax, *out| out.* = try narrow(T, add(T, wide(T, b), negate(T, wide(T, ax))));
        }
        fn restart(self: *Self, value: S.Real(T)) !void {
            try self.local(normalized(T, self.r, self.r, value));
            @memcpy(self.shadow, self.r);
            @memset(self.p, S.zero(T));
            @memset(self.v, S.zero(T));
            self.scale = value;
        }
        fn updateP(self: *Self, beta: Coefficient(T), omega: Coefficient(T)) !void {
            for (self.r, self.p, self.v) |r0, *p0, v0| p0.* = try narrow(T, add(T, wide(T, r0), mul(T, beta, add(T, wide(T, p0.*), negate(T, mul(T, omega, wide(T, v0)))))));
        }
        fn updateS(self: *Self, alpha: Coefficient(T)) !void {
            for (self.r, self.v, self.s) |r0, v0, *s0| s0.* = try narrow(T, add(T, wide(T, r0), negate(T, mul(T, alpha, wide(T, v0)))));
        }
        fn updateX(self: *Self, alpha: Coefficient(T), omega: Coefficient(T)) !void {
            const factor = real(T, self.scale);
            for (self.x, self.phat, self.shat) |*x0, p0, s0| x0.* = try narrow(T, add(T, wide(T, x0.*), mul(T, factor, add(T, mul(T, alpha, wide(T, p0)), mul(T, omega, wide(T, s0))))));
        }
        fn updateR(self: *Self, snorm: S.Real(T), omega: Coefficient(T)) !void {
            for (self.r, self.s, self.t) |*r0, s0, t0| r0.* = try narrow(T, add(T, mul(T, real(T, snorm), wide(T, s0)), negate(T, mul(T, omega, wide(T, t0)))));
        }
        fn done(self: *Self, value: S.Real(T)) bool {
            return @as(Wide(T), value) <= self.target;
        }
        fn estimatedDone(self: *Self, value: S.Real(T)) bool {
            return @as(Wide(T), self.scale) * value <= self.target;
        }
        fn finish(self: *Self, status: Status, residual_norm: S.Real(T)) !Result(T) {
            const selected: Status = if (self.done(residual_norm)) .converged else status;
            try P.check(self.sync(), self.op, try self.point(.stopping_decision, @backingInt(selected)));
            try P.check(self.sync(), self.op, .{ .stage = .commit });
            @memcpy(self.q.output[0..self.x.len], self.x);
            return .{ .status = selected, .iterations = self.iterations, .action_count = self.actions, .preconditioner_count = self.preconditions, .residual_restarts = self.restarts, .residual_norm = residual_norm };
        }
        fn stopped(self: *Self, status: Status) !Result(T) {
            return self.finish(status, try self.residual());
        }
        fn run(self: *Self) !Result(T) {
            const q = self.q;
            if (q.initial) |values| @memcpy(self.x, values) else @memset(self.x, S.zero(T));
            if (q.global_dimension == 0) return self.finish(.converged, 0);
            const bnorm = try self.norm(q.rhs);
            self.target = @max(@as(Wide(T), q.absolute_tolerance), @as(Wide(T), q.relative_tolerance) * bnorm);
            const initial_norm = if (q.initial != null) try self.residual() else blk: {
                @memcpy(self.r, q.rhs);
                break :blk bnorm;
            };
            if (self.done(initial_norm)) return self.finish(.converged, initial_norm);
            if (q.max_iterations == 0) return self.finish(.iteration_limit, initial_norm);
            try self.restart(initial_norm);
            var first = true;
            var previous = real(T, 1);
            var alpha = real(T, 1);
            var omega = real(T, 1);
            while (self.iterations < q.max_iterations) {
                self.iterations += 1;
                const rho = try self.inner(self.shadow, self.r, self.s, self.t);
                if (zero(T, rho)) return self.stopped(.rho_breakdown);
                if (first) @memcpy(self.p, self.r) else {
                    const beta = try self.local(divide(T, mul(T, rho, alpha), mul(T, previous, omega)));
                    try self.local(self.updateP(beta, omega));
                }
                try self.precondition(self.p, self.phat);
                try self.apply(self.phat, self.v, false);
                const denominator = try self.inner(self.shadow, self.v, self.s, self.t);
                if (zero(T, denominator)) return self.stopped(.alpha_breakdown);
                alpha = try self.local(divide(T, rho, denominator));
                try self.local(self.updateS(alpha));
                const snorm = try self.norm(self.s);
                if (self.estimatedDone(snorm)) {
                    // shat is not an operand on the alpha-only branch.
                    @memset(self.shat, S.zero(T));
                    try self.local(self.updateX(alpha, real(T, 0)));
                    const actual = try self.residual();
                    if (self.done(actual)) return self.finish(.converged, actual);
                    if (self.iterations == q.max_iterations) return self.finish(.iteration_limit, actual);
                    self.restarts += 1;
                    try P.check(self.sync(), self.op, try self.point(.residual_restart, self.restarts));
                    try self.restart(actual);
                    first = true;
                    continue;
                }
                try self.precondition(self.s, self.shat);
                try self.apply(self.shat, self.t, false);
                const tnorm = try self.norm(self.t);
                if (tnorm == 0) {
                    try self.local(self.updateX(alpha, real(T, 0)));
                    return self.stopped(.omega_breakdown);
                }
                // r is dead after s is formed. For extreme scales it stores
                // unit t, while s is normalized in place and reconstructed later.
                var s_factor: S.Real(T) = 1;
                if (directInnerRange(T, tnorm, snorm)) {
                    const numerator = try self.innerUnits(self.t, self.s);
                    omega = try self.local(divide(T, wide(T, numerator), real(T, @as(Wide(T), tnorm) * tnorm)));
                } else {
                    try self.local(normalized(T, self.t, self.r, tnorm));
                    try self.local(normalized(T, self.s, self.s, snorm));
                    const gamma = try self.innerUnits(self.r, self.s);
                    omega = try self.local(divide(T, mul(T, wide(T, gamma), real(T, snorm)), real(T, tnorm)));
                    s_factor = snorm;
                }
                if (zero(T, omega)) {
                    try self.local(self.updateX(alpha, real(T, 0)));
                    return self.stopped(.omega_breakdown);
                }
                try self.local(self.updateX(alpha, omega));
                try self.local(self.updateR(s_factor, omega));
                const rnorm = try self.norm(self.r);
                if (self.estimatedDone(rnorm) or self.iterations == q.max_iterations) {
                    const actual = try self.residual();
                    if (self.done(actual)) return self.finish(.converged, actual);
                    if (self.iterations == q.max_iterations) return self.finish(.iteration_limit, actual);
                    self.restarts += 1;
                    try P.check(self.sync(), self.op, try self.point(.residual_restart, self.restarts));
                    try self.restart(actual);
                    first = true;
                    continue;
                }
                previous = rho;
                first = false;
            }
            unreachable;
        }
    };
}
pub fn runAdmitted(comptime T: type, operation: *const execution.context.Operation, q: Request(T), workspace: []T) !Result(T) {
    _ = operation.borrowedContext();
    const sync = q.action.synchronization;
    try P.enter(T, sync, operation, .bicgstab);
    const used = try P.resolve(sync, operation, .{ .stage = .preflight }, preflight(T, q, workspace));
    try P.check(sync, operation, .{ .stage = .global_dimension, .control = @intCast(q.global_dimension) });
    try P.check(sync, operation, .{ .stage = .iteration_limit, .control = @intCast(q.max_iterations) });
    try P.check(sync, operation, .{ .stage = .relative_tolerance, .control = P.realBits(S.Real(T), q.relative_tolerance) });
    try P.check(sync, operation, .{ .stage = .absolute_tolerance, .control = P.realBits(S.Real(T), q.absolute_tolerance) });
    try P.check(sync, operation, .{ .stage = .reduction_backend, .control = @intFromBool(q.reduction != null) });
    try P.check(sync, operation, .{ .stage = .preconditioner_backend, .control = @intFromBool(q.preconditioner != null) });
    try P.check(sync, operation, .{ .stage = .initial_backend, .control = @intFromBool(q.initial != null) });
    try P.scales(T, sync, operation, q.action_scale, q.diagonal_shift);
    const n = q.action.rows;
    var state = State(T){ .op = operation, .q = q, .x = used[0 * n .. 1 * n], .r = used[1 * n .. 2 * n], .shadow = used[2 * n .. 3 * n], .p = used[3 * n .. 4 * n], .v = used[4 * n .. 5 * n], .s = used[5 * n .. 6 * n], .t = used[6 * n .. 7 * n], .phat = used[7 * n .. 8 * n], .shat = used[8 * n .. 9 * n], .work = used[9 * n ..] };
    return state.run();
}
pub fn Managed(comptime E: type) type {
    return struct {
        const C = E.context;
        pub fn runBorrowedAdmitted(comptime T: type, op: *const C.Operation, q: Request(T), workspace: []T) !Result(T) {
            return runAdmitted(T, op, q, workspace);
        }
        pub fn runScratchAdmitted(comptime T: type, op: *const C.Operation, q: Request(T)) !Result(T) {
            _ = op.borrowedContext();
            const sync = q.action.synchronization;
            try P.enter(T, sync, op, .bicgstab);
            try P.resolve(sync, op, .{ .stage = .preflight }, options(T, q));
            const count = try P.resolve(sync, op, .{ .stage = .workspace }, workspaceLength(T, q.action, q.preconditioner));
            var lease = try P.acquire(C, T, sync, op, count, .vector_allocation);
            defer lease.deinit();
            return runAdmitted(T, op, q, lease.lease.values);
        }
        pub fn runBorrowed(comptime T: type, owner: *C.Context, q: Request(T), workspace: []T) !Result(T) {
            var op = try P.begin(q.action.synchronization, owner);
            defer op.end();
            return runAdmitted(T, &op, q, workspace);
        }
        pub fn runScratch(comptime T: type, owner: *C.Context, q: Request(T)) !Result(T) {
            var op = try P.begin(q.action.synchronization, owner);
            defer op.end();
            return runScratchAdmitted(T, &op, q);
        }
    };
}
