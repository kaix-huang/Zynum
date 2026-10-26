// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Optional external callback protocol. All mathematics, storage and admission
//! remain in the injected shared Core. Tokens are borrowed only during callback
//! execution; foreign code must not retain, forge, or use them asynchronously.
const std = @import("std");
pub fn Binding(comptime Core: type, comptime Checked: type) type {
    return struct {
        pub const OwnerIdentity = Core;
        const N = Core.internal.numerics;
        const E = N.Execution;
        const C = E.context;
        const S = N.Scalar;
        const P = N.Synchronization;
        pub const Wire = Checked.Wire;
        pub const Token = opaque {};
        pub const Point = P.Point;
        /// Every participant must bootstrap the same valid hook and entry. The
        /// hook agrees local failures first, then compares the complete Point on
        /// success, and returns one canonical result on every participant.
        pub const Agreement = extern struct {
            context: ?*anyopaque,
            agree: ?*const fn (?*anyopaque, ?*const Token, Point, Wire.result) callconv(.c) Wire.result,
        };
        pub const Action = extern struct {
            context: ?*anyopaque,
            rows: i64,
            columns: i64,
            workspace_elements: i64,
            apply: ?*const fn (?*anyopaque, *const Token, Wire.const_buffer, Wire.buffer, Wire.buffer) callconv(.c) i32,
        };
        /// One synchronous in-place reduction of column-major H followed by S.
        pub const BlockReduction = extern struct {
            context: ?*anyopaque,
            reduce: ?*const fn (?*anyopaque, *const Token, Wire.buffer) callconv(.c) i32,
        };
        pub fn Reduction(comptime T: type) type {
            return extern struct {
                context: ?*anyopaque,
                inner: ?*const fn (?*anyopaque, *const Token, Wire.const_buffer, Wire.const_buffer, *T) callconv(.c) i32,
                norm: ?*const fn (?*anyopaque, *const Token, Wire.const_buffer, *S.Real(T)) callconv(.c) i32,
            };
        }
        pub fn RitzResult(comptime T: type) type {
            return extern struct { status: i32, has_previous_ritz: i32, order: i64, action_count: i64, minimum: S.Real(T), maximum: S.Real(T), min_change: S.Real(T), max_change: S.Real(T), residual_norm: S.Real(T) };
        }
        pub fn ChainResult(comptime T: type) type {
            return extern struct { status: i32, reserved: i32, order: i64, action_count: i64, residual_norm: S.Real(T) };
        }
        pub fn LinearResult(comptime T: type) type {
            return extern struct { status: i32, reserved: i32, iterations: i64, action_count: i64, preconditioner_count: i64, residual_restarts: i64, residual_norm: S.Real(T) };
        }
        pub const INVALID_OPTIONS: i32 = 15;
        pub const ACTION_FAILED: i32 = 16;
        pub const REDUCTION_FAILED: i32 = 17;
        pub const NONFINITE_RESULT: i32 = 18;
        pub const NONHERMITIAN_ACTION: i32 = 19;
        pub const RITZ_FAILED: i32 = 20;
        pub const PHASE_AGREEMENT_FAILED: i32 = 21;
        pub const PHASE_MISMATCH: i32 = 22;
        fn status(err: anyerror) i32 {
            return switch (err) {
                error.InvalidOptions, error.InvalidMetric, error.MissingGlobalReduction, error.InvalidInitialVector => INVALID_OPTIONS,
                error.CallbackActionFailure => ACTION_FAILED,
                error.CallbackReductionFailure => REDUCTION_FAILED,
                error.NonFiniteResult, error.NonFiniteReduction => NONFINITE_RESULT,
                error.NonHermitianAction => NONHERMITIAN_ACTION,
                error.RitzSolverFailure => RITZ_FAILED,
                error.PhaseAgreementFailed => PHASE_AGREEMENT_FAILED,
                error.PhaseMismatch => PHASE_MISMATCH,
                else => Checked.statusCode(err),
            };
        }
        fn failure(err: anyerror, code: i32) Wire.result {
            return .{ .status = status(err), .reserved = 0, .info = code };
        }
        fn contourError(err: anyerror) anyerror {
            return switch (err) {
                error.InvalidInterval, error.InvalidRatio, error.InvalidNodeCount => error.InvalidOptions,
                error.RangeLimit => error.NonFiniteResult,
                else => err,
            };
        }
        fn okay() Wire.result {
            return .{ .status = Wire.OK, .reserved = 0, .info = 0 };
        }
        fn validStatus(code: i32) bool {
            return switch (code) {
                Wire.OK, Wire.INVALID_FLAG, Wire.INVALID_DIMENSION, Wire.INVALID_LEADING_DIMENSION, Wire.BUFFER_TOO_SMALL, Wire.NULL_POINTER, Wire.MISALIGNED_POINTER, Wire.SIZE_OVERFLOW, Wire.DIMENSION_MISMATCH, Wire.ALIASING_NOT_ALLOWED, Wire.INVALID_SPARSE_STRUCTURE, Wire.NONFINITE_INPUT, Wire.NUMERICAL_FAILURE, Wire.INVALID_LENGTH, Wire.INTERNAL_ERROR, INVALID_OPTIONS...PHASE_MISMATCH => true,
                else => false,
            };
        }
        fn shape(comptime T: type, a: Action) !void {
            if (a.rows < 0 or a.columns < 0) return error.InvalidDimension;
            const n = try Checked.countElements(a.rows);
            const m = try Checked.countElements(a.columns);
            const work = try Checked.countElements(a.workspace_elements);
            inline for (.{ n, m, work }) |count| {
                const bytes = std.math.mul(usize, count, @sizeOf(T)) catch return error.SizeOverflow;
                if (bytes > std.math.maxInt(isize)) return error.SizeOverflow;
            }
            if (a.apply == null) return error.NullPointer;
        }
        fn issued(operation: *const C.Operation) *const Token {
            return @ptrCast(operation);
        }
        fn borrowed(token: ?*const Token) !*const C.Operation {
            const ptr = token orelse return error.NullPointer;
            if (@intFromPtr(ptr) % @alignOf(C.Operation) != 0) return error.MisalignedPointer;
            const operation: *const C.Operation = @ptrCast(@alignCast(ptr));
            _ = operation.borrowedContext();
            return operation;
        }
        fn input(comptime T: type, x: []const T) Wire.const_buffer {
            return .{ .data = if (x.len == 0) null else x.ptr, .length = @intCast(x.len) };
        }
        fn output(comptime T: type, x: []T) Wire.buffer {
            return .{ .data = if (x.len == 0) null else x.ptr, .length = @intCast(x.len) };
        }
        fn State(comptime T: type) type {
            return struct {
                action: Action,
                reduction: ?Reduction(T) = null,
                block_reduction: ?BlockReduction = null,
                inverse_action: ?Action = null,
                agreement: ?Agreement = null,
                last_status: std.atomic.Value(i32) = .init(0),
                // Phase callbacks publish one canonical error before unwinding.
                // Use the same explicit publication as callback status codes so
                // optimized catch paths observe the reply and its INFO payload.
                chosen_status: std.atomic.Value(i32) = .init(0),
                chosen_info: std.atomic.Value(i64) = .init(0),
                fn operator(self: *@This()) N.Operator(T) {
                    return .{ .context = self, .rows = @intCast(self.action.rows), .columns = @intCast(self.action.columns), .workspace_elements = @intCast(self.action.workspace_elements), .apply_fn = apply, .synchronization = self.synchronization() };
                }
                fn synchronization(self: *@This()) ?P.Synchronization {
                    return if (self.agreement != null) .{ .context = self, .agree_fn = agree } else null;
                }
                fn contourSynchronization(self: *@This()) ?P.Synchronization {
                    return if (self.agreement != null) .{ .context = self, .agree_fn = contourAgree } else null;
                }
                fn contourOperator(self: *@This()) N.Operator(T) {
                    var action = self.operator();
                    action.synchronization = self.contourSynchronization();
                    return action;
                }
                fn contourAgree(raw: *const anyopaque, operation: ?*const C.Operation, point: Point, local_error: ?anyerror) !void {
                    return agree(raw, operation, point, if (local_error) |err| contourError(err) else null);
                }
                fn agreedFailure(self: *@This(), err: anyerror) Wire.result {
                    const selected = self.chosen_status.load(.acquire);
                    if (selected != 0) return .{ .status = selected, .reserved = 0, .info = self.chosen_info.load(.monotonic) };
                    return failure(err, self.last_status.load(.acquire));
                }
                fn agree(raw: *const anyopaque, operation: ?*const C.Operation, point: Point, local_error: ?anyerror) !void {
                    const self: *@This() = @ptrCast(@alignCast(@constCast(raw)));
                    const local = if (local_error) |err| failure(err, if (err == error.CallbackActionFailure or err == error.CallbackReductionFailure) self.last_status.load(.acquire) else 0) else okay();
                    const descriptor = self.agreement.?;
                    const token: ?*const Token = if (operation) |value| issued(value) else null;
                    const chosen = descriptor.agree.?(descriptor.context, token, point, local);
                    // A successful reply cannot mask a local failure. Malformed
                    // replies terminate the protocol before any public commit.
                    if (chosen.reserved != 0 or !validStatus(chosen.status) or (chosen.status == Wire.OK and (chosen.info != 0 or local_error != null))) {
                        self.chosen_info.store(0, .monotonic);
                        self.chosen_status.store(PHASE_AGREEMENT_FAILED, .release);
                        return error.PhaseAgreementFailed;
                    }
                    if (chosen.status != Wire.OK) {
                        self.chosen_info.store(chosen.info, .monotonic);
                        self.chosen_status.store(chosen.status, .release);
                        return if (chosen.status == PHASE_MISMATCH) error.PhaseMismatch else error.RemotePhaseFailure;
                    }
                }
                fn apply(raw: *const anyopaque, operation: *const C.Operation, x: []const T, y: []T, work: []T) !void {
                    const self: *@This() = @ptrCast(@alignCast(@constCast(raw)));
                    const code = self.action.apply.?(self.action.context, issued(operation), input(T, x), output(T, y), output(T, work));
                    if (code != 0) {
                        self.last_status.store(code, .release);
                        return error.CallbackActionFailure;
                    }
                }
                fn inner(raw: *const anyopaque, operation: *const C.Operation, x: []const T, y: []const T) !T {
                    const self: *@This() = @ptrCast(@alignCast(@constCast(raw)));
                    const r = self.reduction.?;
                    var result: T = undefined;
                    const code = r.inner.?(r.context, issued(operation), input(T, x), input(T, y), &result);
                    if (code != 0) {
                        self.last_status.store(code, .release);
                        return error.CallbackReductionFailure;
                    }
                    return result;
                }
                fn norm(raw: *const anyopaque, operation: *const C.Operation, x: []const T) !S.Real(T) {
                    const self: *@This() = @ptrCast(@alignCast(@constCast(raw)));
                    const r = self.reduction.?;
                    var result: S.Real(T) = undefined;
                    const code = r.norm.?(r.context, issued(operation), input(T, x), &result);
                    if (code != 0) {
                        self.last_status.store(code, .release);
                        return error.CallbackReductionFailure;
                    }
                    return result;
                }
                fn applyInverse(raw: *const anyopaque, operation: *const C.Operation, x: []const T, y: []T, work: []T) !void {
                    const self: *@This() = @ptrCast(@alignCast(@constCast(raw)));
                    const descriptor = self.inverse_action.?;
                    const code = descriptor.apply.?(descriptor.context, issued(operation), input(T, x), output(T, y), output(T, work));
                    if (code != 0) {
                        self.last_status.store(code, .release);
                        return error.CallbackActionFailure;
                    }
                }
                fn inverseOperator(self: *@This()) ?N.Operator(T) {
                    const a = self.inverse_action orelse return null;
                    return .{ .context = self, .rows = @intCast(a.rows), .columns = @intCast(a.columns), .workspace_elements = @intCast(a.workspace_elements), .apply_fn = applyInverse };
                }
                fn blockReduce(raw: *const anyopaque, operation: *const C.Operation, values: []T) !void {
                    const self: *@This() = @ptrCast(@alignCast(@constCast(raw)));
                    const descriptor = self.block_reduction.?;
                    const code = descriptor.reduce.?(descriptor.context, issued(operation), output(T, values));
                    if (code != 0) {
                        self.last_status.store(code, .release);
                        return error.CallbackReductionFailure;
                    }
                }
                fn blockReducer(self: *@This()) ?N.SubspaceProjection.Reduction(T) {
                    return if (self.block_reduction != null) .{ .context = self, .reduce_fn = blockReduce } else null;
                }
                fn reductions(self: *@This()) ?N.Lanczos.Reduction(T) {
                    return if (self.reduction != null) .{ .context = self, .inner_fn = inner, .norm_fn = norm } else null;
                }
            };
        }
        /// Native tests may inject an existing operation. Generated exported wrappers
        /// select beginDefault once and delegate here, then end only their own entry.
        pub fn chebyshevStepAdmitted(comptime T: type, operation: *const C.Operation, a: Action, previous: Wire.const_buffer, current: Wire.const_buffer, next: Wire.buffer) Wire.result {
            _ = operation.borrowedContext();
            shape(T, a) catch |err| return failure(err, 0);
            const old = Checked.readBuffer(T, previous) catch |err| return failure(err, 0);
            const values = Checked.readBuffer(T, current) catch |err| return failure(err, 0);
            const out = Checked.writeBuffer(T, next) catch |err| return failure(err, 0);
            var state = State(T){ .action = a };
            N.Chebyshev.Managed(E).stepScratchAdmitted(T, operation, state.operator(), old, values, out) catch |err| return failure(err, state.last_status.load(.acquire));
            return okay();
        }
        pub fn chebyshevStep(comptime T: type, a: Action, previous: Wire.const_buffer, current: Wire.const_buffer, next: Wire.buffer) Wire.result {
            var operation = E.beginDefault();
            defer operation.end();
            return chebyshevStepAdmitted(T, &operation, a, previous, current, next);
        }
        pub fn chebyshevAdmitted(comptime T: type, operation: *const C.Operation, a: Action, x: Wire.const_buffer, coefficients: Wire.const_buffer, y: Wire.buffer) Wire.result {
            _ = operation.borrowedContext();
            shape(T, a) catch |err| return failure(err, 0);
            const values = Checked.readBuffer(T, x) catch |err| return failure(err, 0);
            const coeff = Checked.readBuffer(T, coefficients) catch |err| return failure(err, 0);
            const out = Checked.writeBuffer(T, y) catch |err| return failure(err, 0);
            var state = State(T){ .action = a };
            N.Chebyshev.Managed(E).runScratchAdmitted(T, operation, .{ .action = state.operator(), .input = values, .coefficients = coeff, .output = out }) catch |err| return failure(err, state.last_status.load(.acquire));
            return okay();
        }
        pub fn chebyshev(comptime T: type, a: Action, x: Wire.const_buffer, coeff: Wire.const_buffer, y: Wire.buffer) Wire.result {
            var op = E.beginDefault();
            defer op.end();
            return chebyshevAdmitted(T, &op, a, x, coeff, y);
        }
        pub fn lanczosAdmitted(comptime T: type, operation: *const C.Operation, a: Action, reduction: ?*const anyopaque, x: Wire.const_buffer, global_dimension: i64, max_order: i64, ritz_tolerance: S.Real(T), hermitian_tolerance: S.Real(T), out: ?*anyopaque) Wire.result {
            _ = operation.borrowedContext();
            shape(T, a) catch |err| return failure(err, 0);
            const raw_target = out orelse return failure(error.NullPointer, 0);
            if (@intFromPtr(raw_target) % @alignOf(RitzResult(T)) != 0) return failure(error.MisalignedPointer, 0);
            const start = @intFromPtr(raw_target);
            const end = std.math.add(usize, start, @sizeOf(RitzResult(T))) catch return failure(error.SizeOverflow, 0);
            const target: *RitzResult(T) = @ptrCast(@alignCast(raw_target));
            const n = Checked.countElements(global_dimension) catch |err| return failure(err, 0);
            const budget = Checked.countElements(max_order) catch |err| return failure(err, 0);
            const initial = Checked.readBuffer(T, x) catch |err| return failure(err, 0);
            var state = State(T){ .action = a };
            if (reduction) |raw_reduction| {
                if (@intFromPtr(raw_reduction) % @alignOf(Reduction(T)) != 0) return failure(error.MisalignedPointer, 0);
                const r_start = @intFromPtr(raw_reduction);
                const r_end = std.math.add(usize, r_start, @sizeOf(Reduction(T))) catch return failure(error.SizeOverflow, 0);
                if (start < r_end and r_start < end) return failure(error.AliasingNotAllowed, 0);
                const r: *const Reduction(T) = @ptrCast(@alignCast(raw_reduction));
                if (r.inner == null or r.norm == null) return failure(error.NullPointer, 0);
                state.reduction = r.*;
            }
            // The scalar result is committed last; reject aliases against borrowed input
            // before its publication. The opaque callback state retains caller obligations.
            const bytes = std.math.mul(usize, initial.len, @sizeOf(T)) catch return failure(error.SizeOverflow, 0);
            const first = @intFromPtr(initial.ptr);
            const last = std.math.add(usize, first, bytes) catch return failure(error.SizeOverflow, 0);
            if (bytes != 0 and start < last and first < end) return failure(error.AliasingNotAllowed, 0);
            const result = N.Lanczos.Managed(E).runScratchAdmitted(T, operation, .{ .action = state.operator(), .initial = initial, .global_dimension = n, .max_order = budget, .ritz_tolerance = ritz_tolerance, .hermitian_tolerance = hermitian_tolerance, .reduction = state.reductions() }) catch |err| return failure(err, state.last_status.load(.acquire));
            target.* = .{ .status = switch (result.status) {
                .converged => 0,
                .exhausted_dimension => 1,
                .early_breakdown => 2,
                .iteration_limit => 3,
            }, .has_previous_ritz = if (result.has_previous_ritz) 1 else 0, .order = @intCast(result.order), .action_count = @intCast(result.action_count), .minimum = result.minimum, .maximum = result.maximum, .min_change = result.min_change, .max_change = result.max_change, .residual_norm = result.residual_norm };
            return okay();
        }
        pub fn lanczos(comptime T: type, a: Action, r: ?*const anyopaque, x: Wire.const_buffer, n: i64, budget: i64, tol: S.Real(T), hermitian_tol: S.Real(T), out: ?*anyopaque) Wire.result {
            var op = E.beginDefault();
            defer op.end();
            return lanczosAdmitted(T, &op, a, r, x, n, budget, tol, hermitian_tol, out);
        }
        fn chainRange(values: anytype) !N.Storage.ByteRange {
            return N.Storage.byteRange(@intFromPtr(values.ptr), try N.Storage.byteLength(values.len, @sizeOf(@TypeOf(values[0]))));
        }
        /// Public coefficients and summary commit only after every synchronous
        /// callback succeeds; callback state must not mutate borrowed outputs.
        pub fn lanczosChainAdmitted(comptime T: type, operation: *const C.Operation, a: Action, reduction: ?*const anyopaque, x: Wire.const_buffer, global_dimension: i64, max_order: i64, breakdown_tolerance: S.Real(T), hermitian_tolerance: S.Real(T), alpha_buffer: Wire.buffer, beta_buffer: Wire.buffer, out: ?*anyopaque) Wire.result {
            _ = operation.borrowedContext();
            shape(T, a) catch |err| return failure(err, 0);
            const raw_target = out orelse return failure(error.NullPointer, 0);
            if (@intFromPtr(raw_target) % @alignOf(ChainResult(T)) != 0) return failure(error.MisalignedPointer, 0);
            const target_range = N.Storage.byteRange(@intFromPtr(raw_target), @sizeOf(ChainResult(T))) catch |err| return failure(err, 0);
            const target: *ChainResult(T) = @ptrCast(@alignCast(raw_target));
            const n = Checked.countElements(global_dimension) catch |err| return failure(err, 0);
            const budget = Checked.countElements(max_order) catch |err| return failure(err, 0);
            const initial = Checked.readBuffer(T, x) catch |err| return failure(err, 0);
            const alpha = Checked.writeBuffer(S.Real(T), alpha_buffer) catch |err| return failure(err, 0);
            const beta = Checked.writeBuffer(S.Real(T), beta_buffer) catch |err| return failure(err, 0);
            const input_range = chainRange(initial) catch |err| return failure(err, 0);
            const alpha_range = chainRange(alpha) catch |err| return failure(err, 0);
            const beta_range = chainRange(beta) catch |err| return failure(err, 0);
            const ranges = [_]N.Storage.ByteRange{ input_range, alpha_range, beta_range, target_range };
            for (ranges, 0..) |first, i| {
                for (ranges[i + 1 ..]) |second| {
                    if (N.Storage.overlaps(first, second)) return failure(error.AliasingNotAllowed, 0);
                }
            }
            var state = State(T){ .action = a };
            if (reduction) |raw_reduction| {
                if (@intFromPtr(raw_reduction) % @alignOf(Reduction(T)) != 0) return failure(error.MisalignedPointer, 0);
                const reduction_range = N.Storage.byteRange(@intFromPtr(raw_reduction), @sizeOf(Reduction(T))) catch |err| return failure(err, 0);
                for ([_]N.Storage.ByteRange{ alpha_range, beta_range, target_range }) |output_range| {
                    if (N.Storage.overlaps(reduction_range, output_range)) return failure(error.AliasingNotAllowed, 0);
                }
                const r: *const Reduction(T) = @ptrCast(@alignCast(raw_reduction));
                if (r.inner == null or r.norm == null) return failure(error.NullPointer, 0);
                state.reduction = r.*;
            }
            const result = N.LanczosChain.Managed(E).runScratchAdmitted(T, operation, .{ .action = state.operator(), .initial = initial, .global_dimension = n, .max_order = budget, .breakdown_tolerance = breakdown_tolerance, .hermitian_tolerance = hermitian_tolerance, .reduction = state.reductions(), .alpha = alpha, .beta = beta }) catch |err| return failure(err, state.last_status.load(.acquire));
            target.* = .{ .status = switch (result.status) {
                .depth_limit => 0,
                .breakdown => 1,
            }, .reserved = 0, .order = @intCast(result.order), .action_count = @intCast(result.action_count), .residual_norm = result.residual_norm };
            return okay();
        }
        pub fn lanczosChain(comptime T: type, a: Action, r: ?*const anyopaque, x: Wire.const_buffer, n: i64, budget: i64, tol: S.Real(T), hermitian_tol: S.Real(T), alpha: Wire.buffer, beta: Wire.buffer, out: ?*anyopaque) Wire.result {
            var operation = E.beginDefault();
            defer operation.end();
            return lanczosChainAdmitted(T, &operation, a, r, x, n, budget, tol, hermitian_tol, alpha, beta, out);
        }
        fn rejectOverlap(first: N.Storage.ByteRange, second: N.Storage.ByteRange) !void {
            if (N.Storage.overlaps(first, second)) return error.AliasingNotAllowed;
        }
        fn summaryTarget(comptime T: type, raw: ?*anyopaque) !*T {
            const target = raw orelse return error.NullPointer;
            if (@intFromPtr(target) % @alignOf(T) != 0) return error.MisalignedPointer;
            _ = try N.Storage.byteRange(@intFromPtr(target), @sizeOf(T));
            return @ptrCast(@alignCast(target));
        }
        fn decodedReduction(comptime T: type, raw: ?*const anyopaque, outputs: []const N.Storage.ByteRange) !?Reduction(T) {
            const pointer = raw orelse return null;
            if (@intFromPtr(pointer) % @alignOf(Reduction(T)) != 0) return error.MisalignedPointer;
            const range = try N.Storage.byteRange(@intFromPtr(pointer), @sizeOf(Reduction(T)));
            for (outputs) |out| try rejectOverlap(range, out);
            const reduction: *const Reduction(T) = @ptrCast(@alignCast(pointer));
            if (reduction.inner == null or reduction.norm == null) return error.NullPointer;
            return reduction.*;
        }
        fn StepArguments(comptime T: type) type {
            return struct { previous: []const T, current: []const T, next: []T };
        }
        fn decodeStep(comptime T: type, a: Action, previous: Wire.const_buffer, current: Wire.const_buffer, next: Wire.buffer) !StepArguments(T) {
            try shape(T, a);
            const old = try Checked.readBuffer(T, previous);
            const values = try Checked.readBuffer(T, current);
            const out = try Checked.writeBuffer(T, next);
            const n: usize = @intCast(a.rows);
            if (a.rows != a.columns or old.len != n or values.len != n or out.len != n) return error.DimensionMismatch;
            const out_range = try chainRange(out);
            try rejectOverlap(try chainRange(old), out_range);
            try rejectOverlap(try chainRange(values), out_range);
            return .{ .previous = old, .current = values, .next = out };
        }
        fn SeriesArguments(comptime T: type) type {
            return struct { input: []const T, coefficients: []const T, output: []T };
        }
        fn decodeSeries(comptime T: type, a: Action, x: Wire.const_buffer, coefficients: Wire.const_buffer, y: Wire.buffer) !SeriesArguments(T) {
            try shape(T, a);
            const values = try Checked.readBuffer(T, x);
            const coeff = try Checked.readBuffer(T, coefficients);
            const out = try Checked.writeBuffer(T, y);
            const n: usize = @intCast(a.rows);
            if (a.rows != a.columns or values.len != n or out.len != n) return error.DimensionMismatch;
            if (coeff.len == 0) return error.InvalidOptions;
            const out_range = try chainRange(out);
            try rejectOverlap(try chainRange(values), out_range);
            try rejectOverlap(try chainRange(coeff), out_range);
            return .{ .input = values, .coefficients = coeff, .output = out };
        }
        fn reductionOptions(comptime T: type, a: Action, initial: []const T, n: usize, budget: usize, tolerance: S.Real(T), hermitian_tolerance: S.Real(T), reduction: ?Reduction(T)) !void {
            const rows: usize = @intCast(a.rows);
            if (a.rows != a.columns or initial.len != rows) return error.DimensionMismatch;
            if (n == 0 or rows > n or budget == 0 or !std.math.isFinite(tolerance) or tolerance < 0 or !std.math.isFinite(hermitian_tolerance) or hermitian_tolerance < 0) return error.InvalidOptions;
            if (reduction == null and n != rows) return error.MissingGlobalReduction;
        }
        fn RitzArguments(comptime T: type) type {
            return struct { target: *RitzResult(T), initial: []const T, dimension: usize, budget: usize, reduction: ?Reduction(T) };
        }
        fn decodeRitz(comptime T: type, a: Action, raw_reduction: ?*const anyopaque, x: Wire.const_buffer, global_dimension: i64, max_order: i64, ritz_tolerance: S.Real(T), hermitian_tolerance: S.Real(T), out: ?*anyopaque) !RitzArguments(T) {
            try shape(T, a);
            const target = try summaryTarget(RitzResult(T), out);
            const target_range = try N.Storage.byteRange(@intFromPtr(target), @sizeOf(RitzResult(T)));
            const n = try Checked.countElements(global_dimension);
            const budget = try Checked.countElements(max_order);
            const initial = try Checked.readBuffer(T, x);
            const reduction = try decodedReduction(T, raw_reduction, &.{target_range});
            try rejectOverlap(try chainRange(initial), target_range);
            try reductionOptions(T, a, initial, n, budget, ritz_tolerance, hermitian_tolerance, reduction);
            return .{ .target = target, .initial = initial, .dimension = n, .budget = budget, .reduction = reduction };
        }
        fn ChainArguments(comptime T: type) type {
            return struct { target: *ChainResult(T), initial: []const T, dimension: usize, budget: usize, reduction: ?Reduction(T), alpha: []S.Real(T), beta: []S.Real(T) };
        }
        fn decodeChain(comptime T: type, a: Action, raw_reduction: ?*const anyopaque, x: Wire.const_buffer, global_dimension: i64, max_order: i64, breakdown_tolerance: S.Real(T), hermitian_tolerance: S.Real(T), alpha_buffer: Wire.buffer, beta_buffer: Wire.buffer, out: ?*anyopaque) !ChainArguments(T) {
            try shape(T, a);
            const target = try summaryTarget(ChainResult(T), out);
            const target_range = try N.Storage.byteRange(@intFromPtr(target), @sizeOf(ChainResult(T)));
            const n = try Checked.countElements(global_dimension);
            const budget = try Checked.countElements(max_order);
            const initial = try Checked.readBuffer(T, x);
            const alpha = try Checked.writeBuffer(S.Real(T), alpha_buffer);
            const beta = try Checked.writeBuffer(S.Real(T), beta_buffer);
            const alpha_range = try chainRange(alpha);
            const beta_range = try chainRange(beta);
            // Full declared spans, including unused tails, participate in alias
            // validation before any callback or transactional publication.
            const ranges = [_]N.Storage.ByteRange{ try chainRange(initial), alpha_range, beta_range, target_range };
            for (ranges, 0..) |first, i| {
                for (ranges[i + 1 ..]) |second| try rejectOverlap(first, second);
            }
            const reduction = try decodedReduction(T, raw_reduction, &.{ alpha_range, beta_range, target_range });
            try reductionOptions(T, a, initial, n, budget, breakdown_tolerance, hermitian_tolerance, reduction);
            if (alpha.len < budget or beta.len < budget - 1) return error.BufferTooSmall;
            return .{ .target = target, .initial = initial, .dimension = n, .budget = budget, .reduction = reduction, .alpha = alpha, .beta = beta };
        }
        /// Agreed entries borrow the original operation and external hook. Wire
        /// decoding is one local outcome, resolved before any numerical callback.
        /// A missing hook violates bootstrap and cannot be repaired collectively.
        pub fn chebyshevStepAgreedAdmitted(comptime T: type, operation: *const C.Operation, a: Action, agreement: Agreement, previous: Wire.const_buffer, current: Wire.const_buffer, next: Wire.buffer) Wire.result {
            _ = operation.borrowedContext();
            if (agreement.agree == null) return failure(error.PhaseAgreementFailed, 0);
            var state = State(T){ .action = a, .agreement = agreement };
            const sync = state.synchronization();
            P.enter(T, sync, operation, .chebyshev_step) catch |err| return state.agreedFailure(err);
            const arguments = P.resolve(sync, operation, .{ .stage = .preflight }, decodeStep(T, a, previous, current, next)) catch |err| return state.agreedFailure(err);
            N.Chebyshev.Managed(E).stepScratchAdmitted(T, operation, state.operator(), arguments.previous, arguments.current, arguments.next) catch |err| return state.agreedFailure(err);
            return okay();
        }
        pub fn chebyshevStepAgreed(comptime T: type, a: Action, agreement: Agreement, previous: Wire.const_buffer, current: Wire.const_buffer, next: Wire.buffer) Wire.result {
            var operation = E.beginDefault();
            defer operation.end();
            return chebyshevStepAgreedAdmitted(T, &operation, a, agreement, previous, current, next);
        }
        pub fn chebyshevAgreedAdmitted(comptime T: type, operation: *const C.Operation, a: Action, agreement: Agreement, x: Wire.const_buffer, coefficients: Wire.const_buffer, y: Wire.buffer) Wire.result {
            _ = operation.borrowedContext();
            if (agreement.agree == null) return failure(error.PhaseAgreementFailed, 0);
            var state = State(T){ .action = a, .agreement = agreement };
            const sync = state.synchronization();
            P.enter(T, sync, operation, .chebyshev_series) catch |err| return state.agreedFailure(err);
            const arguments = P.resolve(sync, operation, .{ .stage = .preflight }, decodeSeries(T, a, x, coefficients, y)) catch |err| return state.agreedFailure(err);
            N.Chebyshev.Managed(E).runScratchAdmitted(T, operation, .{ .action = state.operator(), .input = arguments.input, .coefficients = arguments.coefficients, .output = arguments.output }) catch |err| return state.agreedFailure(err);
            return okay();
        }
        pub fn chebyshevAgreed(comptime T: type, a: Action, agreement: Agreement, x: Wire.const_buffer, coefficients: Wire.const_buffer, y: Wire.buffer) Wire.result {
            var operation = E.beginDefault();
            defer operation.end();
            return chebyshevAgreedAdmitted(T, &operation, a, agreement, x, coefficients, y);
        }
        pub fn lanczosAgreedAdmitted(comptime T: type, operation: *const C.Operation, a: Action, agreement: Agreement, reduction: ?*const anyopaque, x: Wire.const_buffer, global_dimension: i64, max_order: i64, ritz_tolerance: S.Real(T), hermitian_tolerance: S.Real(T), out: ?*anyopaque) Wire.result {
            _ = operation.borrowedContext();
            if (agreement.agree == null) return failure(error.PhaseAgreementFailed, 0);
            var state = State(T){ .action = a, .agreement = agreement };
            const sync = state.synchronization();
            P.enter(T, sync, operation, .lanczos_ritz) catch |err| return state.agreedFailure(err);
            const arguments = P.resolve(sync, operation, .{ .stage = .preflight }, decodeRitz(T, a, reduction, x, global_dimension, max_order, ritz_tolerance, hermitian_tolerance, out)) catch |err| return state.agreedFailure(err);
            state.reduction = arguments.reduction;
            const result = N.Lanczos.Managed(E).runScratchAdmitted(T, operation, .{ .action = state.operator(), .initial = arguments.initial, .global_dimension = arguments.dimension, .max_order = arguments.budget, .ritz_tolerance = ritz_tolerance, .hermitian_tolerance = hermitian_tolerance, .reduction = state.reductions() }) catch |err| return state.agreedFailure(err);
            arguments.target.* = .{ .status = switch (result.status) {
                .converged => 0,
                .exhausted_dimension => 1,
                .early_breakdown => 2,
                .iteration_limit => 3,
            }, .has_previous_ritz = if (result.has_previous_ritz) 1 else 0, .order = @intCast(result.order), .action_count = @intCast(result.action_count), .minimum = result.minimum, .maximum = result.maximum, .min_change = result.min_change, .max_change = result.max_change, .residual_norm = result.residual_norm };
            return okay();
        }
        pub fn lanczosAgreed(comptime T: type, a: Action, agreement: Agreement, reduction: ?*const anyopaque, x: Wire.const_buffer, global_dimension: i64, max_order: i64, ritz_tolerance: S.Real(T), hermitian_tolerance: S.Real(T), out: ?*anyopaque) Wire.result {
            var operation = E.beginDefault();
            defer operation.end();
            return lanczosAgreedAdmitted(T, &operation, a, agreement, reduction, x, global_dimension, max_order, ritz_tolerance, hermitian_tolerance, out);
        }
        pub fn lanczosChainAgreedAdmitted(comptime T: type, operation: *const C.Operation, a: Action, agreement: Agreement, reduction: ?*const anyopaque, x: Wire.const_buffer, global_dimension: i64, max_order: i64, breakdown_tolerance: S.Real(T), hermitian_tolerance: S.Real(T), alpha_buffer: Wire.buffer, beta_buffer: Wire.buffer, out: ?*anyopaque) Wire.result {
            _ = operation.borrowedContext();
            if (agreement.agree == null) return failure(error.PhaseAgreementFailed, 0);
            var state = State(T){ .action = a, .agreement = agreement };
            const sync = state.synchronization();
            P.enter(T, sync, operation, .lanczos_chain) catch |err| return state.agreedFailure(err);
            const arguments = P.resolve(sync, operation, .{ .stage = .preflight }, decodeChain(T, a, reduction, x, global_dimension, max_order, breakdown_tolerance, hermitian_tolerance, alpha_buffer, beta_buffer, out)) catch |err| return state.agreedFailure(err);
            state.reduction = arguments.reduction;
            const result = N.LanczosChain.Managed(E).runScratchAdmitted(T, operation, .{ .action = state.operator(), .initial = arguments.initial, .global_dimension = arguments.dimension, .max_order = arguments.budget, .breakdown_tolerance = breakdown_tolerance, .hermitian_tolerance = hermitian_tolerance, .reduction = state.reductions(), .alpha = arguments.alpha, .beta = arguments.beta }) catch |err| return state.agreedFailure(err);
            arguments.target.* = .{ .status = switch (result.status) {
                .depth_limit => 0,
                .breakdown => 1,
            }, .reserved = 0, .order = @intCast(result.order), .action_count = @intCast(result.action_count), .residual_norm = result.residual_norm };
            return okay();
        }
        pub fn lanczosChainAgreed(comptime T: type, a: Action, agreement: Agreement, reduction: ?*const anyopaque, x: Wire.const_buffer, global_dimension: i64, max_order: i64, breakdown_tolerance: S.Real(T), hermitian_tolerance: S.Real(T), alpha: Wire.buffer, beta: Wire.buffer, out: ?*anyopaque) Wire.result {
            var operation = E.beginDefault();
            defer operation.end();
            return lanczosChainAgreedAdmitted(T, &operation, a, agreement, reduction, x, global_dimension, max_order, breakdown_tolerance, hermitian_tolerance, alpha, beta, out);
        }
        /// Additive callback-panel wire operands. Existing checked dense ABI is
        /// unchanged; layouts belong to this frontend and decode into shared Core.
        pub const ConstPanel = extern struct { dense: Wire.const_dense, layout: i32, reserved: i32 = 0 };
        pub const Panel = extern struct { dense: Wire.dense, layout: i32, reserved: i32 = 0 };
        fn panelShape(dense: anytype, layout: i32, reserved: i32) !N.Dense.PanelShape {
            if (reserved != 0) return error.InvalidFlag;
            return .{ .row_count = try Checked.countElements(dense.rows), .column_count = try Checked.countElements(dense.cols), .leading_dimension = try Checked.countElements(dense.ld), .layout = switch (layout) {
                101 => .row_major,
                102 => .column_major,
                else => return error.InvalidFlag,
            } };
        }
        fn FilterArguments(comptime T: type) type {
            return struct { input: N.Dense.ConstPanel(T), output: N.Dense.Panel(T), degree: usize, bounds: N.NormalizedChebyshev.Bounds(T) };
        }
        fn decodeFilter(comptime T: type, a: Action, x: ConstPanel, degree: i64, lower: S.Real(T), cutoff: S.Real(T), upper: S.Real(T), y: Panel) !FilterArguments(T) {
            try shape(T, a);
            if (degree < 0) return error.InvalidOptions;
            const order = try Checked.countElements(degree);
            const input_panel: N.Dense.ConstPanel(T) = .{ .values = try Checked.readBuffer(T, x.dense.storage), .shape = try panelShape(x.dense, x.layout, x.reserved) };
            const output_panel: N.Dense.Panel(T) = .{ .values = try Checked.writeBuffer(T, y.dense.storage), .shape = try panelShape(y.dense, y.layout, y.reserved) };
            _ = try input_panel.check();
            _ = try output_panel.check();
            if (a.rows != a.columns or input_panel.shape.row_count != @as(usize, @intCast(a.columns)) or output_panel.shape.row_count != @as(usize, @intCast(a.rows)) or input_panel.shape.column_count != output_panel.shape.column_count) return error.DimensionMismatch;
            if (order > 0) {
                if (!std.math.isFinite(lower) or !std.math.isFinite(cutoff) or !std.math.isFinite(upper)) return error.NonFiniteInput;
                if (!(lower < cutoff and cutoff < upper)) return error.InvalidOptions;
            }
            return .{ .input = input_panel, .output = output_panel, .degree = order, .bounds = .{ .lower = lower, .cutoff = cutoff, .upper = upper } };
        }
        fn vectorInput(x: Wire.const_buffer) ConstPanel {
            return .{ .dense = .{ .storage = x, .rows = x.length, .cols = 1, .ld = x.length }, .layout = 102 };
        }
        fn vectorOutput(y: Wire.buffer) Panel {
            return .{ .dense = .{ .storage = y, .rows = y.length, .cols = 1, .ld = y.length }, .layout = 102 };
        }
        pub fn normalizedFilterPanelAdmitted(comptime T: type, operation: *const C.Operation, a: Action, x: ConstPanel, degree: i64, lower: S.Real(T), cutoff: S.Real(T), upper: S.Real(T), y: Panel) Wire.result {
            _ = operation.borrowedContext();
            const arguments = decodeFilter(T, a, x, degree, lower, cutoff, upper, y) catch |err| return failure(err, 0);
            var state = State(T){ .action = a };
            N.NormalizedChebyshev.Managed(E).runPanelScratchAdmitted(T, operation, .{ .action = state.operator(), .input = arguments.input, .output = arguments.output, .degree = arguments.degree, .bounds = arguments.bounds }) catch |err| return failure(err, state.last_status.load(.acquire));
            return okay();
        }
        pub fn normalizedFilterPanel(comptime T: type, a: Action, x: ConstPanel, degree: i64, lower: S.Real(T), cutoff: S.Real(T), upper: S.Real(T), y: Panel) Wire.result {
            var operation = E.beginDefault();
            defer operation.end();
            return normalizedFilterPanelAdmitted(T, &operation, a, x, degree, lower, cutoff, upper, y);
        }
        pub fn normalizedFilterAdmitted(comptime T: type, operation: *const C.Operation, a: Action, x: Wire.const_buffer, degree: i64, lower: S.Real(T), cutoff: S.Real(T), upper: S.Real(T), y: Wire.buffer) Wire.result {
            return normalizedFilterPanelAdmitted(T, operation, a, vectorInput(x), degree, lower, cutoff, upper, vectorOutput(y));
        }
        pub fn normalizedFilter(comptime T: type, a: Action, x: Wire.const_buffer, degree: i64, lower: S.Real(T), cutoff: S.Real(T), upper: S.Real(T), y: Wire.buffer) Wire.result {
            var operation = E.beginDefault();
            defer operation.end();
            return normalizedFilterAdmitted(T, &operation, a, x, degree, lower, cutoff, upper, y);
        }
        pub fn normalizedFilterPanelAgreedAdmitted(comptime T: type, operation: *const C.Operation, a: Action, agreement: Agreement, x: ConstPanel, degree: i64, lower: S.Real(T), cutoff: S.Real(T), upper: S.Real(T), y: Panel) Wire.result {
            _ = operation.borrowedContext();
            if (agreement.agree == null) return failure(error.PhaseAgreementFailed, 0);
            var state = State(T){ .action = a, .agreement = agreement };
            const sync = state.synchronization();
            P.enter(T, sync, operation, .normalized_chebyshev) catch |err| return state.agreedFailure(err);
            const arguments = P.resolve(sync, operation, .{ .stage = .preflight }, decodeFilter(T, a, x, degree, lower, cutoff, upper, y)) catch |err| return state.agreedFailure(err);
            N.NormalizedChebyshev.Managed(E).runPanelScratchAdmitted(T, operation, .{ .action = state.operator(), .input = arguments.input, .output = arguments.output, .degree = arguments.degree, .bounds = arguments.bounds }) catch |err| return state.agreedFailure(err);
            return okay();
        }
        pub fn normalizedFilterPanelAgreed(comptime T: type, a: Action, agreement: Agreement, x: ConstPanel, degree: i64, lower: S.Real(T), cutoff: S.Real(T), upper: S.Real(T), y: Panel) Wire.result {
            var operation = E.beginDefault();
            defer operation.end();
            return normalizedFilterPanelAgreedAdmitted(T, &operation, a, agreement, x, degree, lower, cutoff, upper, y);
        }
        pub fn normalizedFilterAgreedAdmitted(comptime T: type, operation: *const C.Operation, a: Action, agreement: Agreement, x: Wire.const_buffer, degree: i64, lower: S.Real(T), cutoff: S.Real(T), upper: S.Real(T), y: Wire.buffer) Wire.result {
            return normalizedFilterPanelAgreedAdmitted(T, operation, a, agreement, vectorInput(x), degree, lower, cutoff, upper, vectorOutput(y));
        }
        pub fn normalizedFilterAgreed(comptime T: type, a: Action, agreement: Agreement, x: Wire.const_buffer, degree: i64, lower: S.Real(T), cutoff: S.Real(T), upper: S.Real(T), y: Wire.buffer) Wire.result {
            var operation = E.beginDefault();
            defer operation.end();
            return normalizedFilterAgreedAdmitted(T, &operation, a, agreement, x, degree, lower, cutoff, upper, y);
        }

        fn decodeProjection(comptime T: type, y: ConstPanel, z: ConstPanel, weighted: i32, weights: Wire.const_buffer, h: Panel, s: Panel) !N.SubspaceProjection.Request(T) {
            if (weighted != 0 and weighted != 1) return error.InvalidOptions;
            const request: N.SubspaceProjection.Request(T) = .{
                .basis = .{ .values = try Checked.readBuffer(T, y.dense.storage), .shape = try panelShape(y.dense, y.layout, y.reserved) },
                .applied = .{ .values = try Checked.readBuffer(T, z.dense.storage), .shape = try panelShape(z.dense, z.layout, z.reserved) },
                // An explicit mode preserves weighted empty shards. Disabled
                // weights are ignored, including their wire descriptor.
                .weights = if (weighted == 1) try Checked.readBuffer(S.Real(T), weights) else null,
                .hamiltonian = .{ .values = try Checked.writeBuffer(T, h.dense.storage), .shape = try panelShape(h.dense, h.layout, h.reserved) },
                .overlap = .{ .values = try Checked.writeBuffer(T, s.dense.storage), .shape = try panelShape(s.dense, s.layout, s.reserved) },
            };
            _ = try request.basis.check();
            _ = try request.applied.check();
            _ = try request.hamiltonian.check();
            _ = try request.overlap.check();
            return request;
        }
        fn subspaceProjectEntryAdmitted(comptime T: type, operation: *const C.Operation, reduction: BlockReduction, agreement: ?Agreement, y: ConstPanel, z: ConstPanel, weighted: i32, weights: Wire.const_buffer, h: Panel, s: Panel) Wire.result {
            _ = operation.borrowedContext();
            if (agreement) |hook| if (hook.agree == null) return failure(error.PhaseAgreementFailed, 0);
            var state = State(T){ .action = .{ .context = null, .rows = 0, .columns = 0, .workspace_elements = 0, .apply = null }, .agreement = agreement, .block_reduction = if (reduction.reduce != null) reduction else null };
            const sync = state.synchronization();
            P.enter(T, sync, operation, .subspace_projection) catch |err| return state.agreedFailure(err);
            var request = P.resolve(sync, operation, .{ .stage = .preflight }, decodeProjection(T, y, z, weighted, weights, h, s)) catch |err| return state.agreedFailure(err);
            request.reduction = state.blockReducer();
            request.synchronization = sync;
            N.SubspaceProjection.Managed(E).runScratchAdmitted(T, operation, request) catch |err| return state.agreedFailure(err);
            return okay();
        }
        pub fn subspaceProjectAdmitted(comptime T: type, operation: *const C.Operation, reduction: BlockReduction, y: ConstPanel, z: ConstPanel, weighted: i32, weights: Wire.const_buffer, h: Panel, s: Panel) Wire.result {
            return subspaceProjectEntryAdmitted(T, operation, reduction, null, y, z, weighted, weights, h, s);
        }
        pub fn subspaceProject(comptime T: type, reduction: BlockReduction, y: ConstPanel, z: ConstPanel, weighted: i32, weights: Wire.const_buffer, h: Panel, s: Panel) Wire.result {
            var operation = E.beginDefault();
            defer operation.end();
            return subspaceProjectAdmitted(T, &operation, reduction, y, z, weighted, weights, h, s);
        }
        pub fn subspaceProjectAgreedAdmitted(comptime T: type, operation: *const C.Operation, reduction: BlockReduction, agreement: Agreement, y: ConstPanel, z: ConstPanel, weighted: i32, weights: Wire.const_buffer, h: Panel, s: Panel) Wire.result {
            return subspaceProjectEntryAdmitted(T, operation, reduction, agreement, y, z, weighted, weights, h, s);
        }
        pub fn subspaceProjectAgreed(comptime T: type, reduction: BlockReduction, agreement: Agreement, y: ConstPanel, z: ConstPanel, weighted: i32, weights: Wire.const_buffer, h: Panel, s: Panel) Wire.result {
            var operation = E.beginDefault();
            defer operation.end();
            return subspaceProjectAgreedAdmitted(T, &operation, reduction, agreement, y, z, weighted, weights, h, s);
        }

        fn SeriesSetArguments(comptime T: type) type {
            return struct { input: []const T, coefficients: N.Dense.ConstPanel(T), counts: []const i64, output: N.Dense.Panel(T) };
        }
        fn decodeSeriesSet(comptime T: type, a: Action, x: Wire.const_buffer, coefficients: ConstPanel, counts: Wire.const_buffer, y: Panel) !SeriesSetArguments(T) {
            try shape(T, a);
            const values = try Checked.readBuffer(T, x);
            const table: N.Dense.ConstPanel(T) = .{ .values = try Checked.readBuffer(T, coefficients.dense.storage), .shape = try panelShape(coefficients.dense, coefficients.layout, coefficients.reserved) };
            const lengths = try Checked.readBuffer(i64, counts);
            const target: N.Dense.Panel(T) = .{ .values = try Checked.writeBuffer(T, y.dense.storage), .shape = try panelShape(y.dense, y.layout, y.reserved) };
            _ = try table.check();
            const used = try target.check();
            if (a.rows != a.columns or values.len != @as(usize, @intCast(a.columns)) or target.shape.row_count != @as(usize, @intCast(a.rows)) or table.shape.column_count != lengths.len or target.shape.column_count != lengths.len) return error.DimensionMismatch;
            for (lengths) |length| {
                if (length <= 0) return error.InvalidOptions;
                if (try Checked.countElements(length) > table.shape.row_count) return error.InvalidOptions;
            }
            // Native lengths are copied into captured Context scratch below.
            // Preserve the public read-only count buffer through final commit.
            try rejectOverlap(try chainRange(lengths), try chainRange(target.values[0..used]));
            return .{ .input = values, .coefficients = table, .counts = lengths, .output = target };
        }
        fn seriesSetEntryAdmitted(comptime T: type, operation: *const C.Operation, a: Action, agreement: ?Agreement, x: Wire.const_buffer, coefficients: ConstPanel, counts: Wire.const_buffer, y: Panel) Wire.result {
            _ = operation.borrowedContext();
            if (agreement) |hook| if (hook.agree == null) return failure(error.PhaseAgreementFailed, 0);
            var state = State(T){ .action = a, .agreement = agreement };
            const sync = state.synchronization();
            P.enter(T, sync, operation, .chebyshev_series_set) catch |err| return state.agreedFailure(err);
            const arguments = P.resolve(sync, operation, .{ .stage = .preflight }, decodeSeriesSet(T, a, x, coefficients, counts, y)) catch |err| return state.agreedFailure(err);
            var lengths = P.acquire(C, usize, sync, operation, arguments.counts.len, .vector_allocation) catch |err| return state.agreedFailure(err);
            defer lengths.deinit();
            for (arguments.counts, lengths.lease.values) |length, *native| native.* = @intCast(length);
            N.Chebyshev.Managed(E).runSeriesSetScratchAdmitted(T, operation, .{ .action = state.operator(), .input = arguments.input, .coefficients = arguments.coefficients, .coefficient_counts = lengths.lease.values, .output = arguments.output }) catch |err| return state.agreedFailure(err);
            return okay();
        }
        pub fn chebyshevSeriesSetAdmitted(comptime T: type, operation: *const C.Operation, a: Action, x: Wire.const_buffer, coefficients: ConstPanel, counts: Wire.const_buffer, y: Panel) Wire.result {
            return seriesSetEntryAdmitted(T, operation, a, null, x, coefficients, counts, y);
        }
        pub fn chebyshevSeriesSet(comptime T: type, a: Action, x: Wire.const_buffer, coefficients: ConstPanel, counts: Wire.const_buffer, y: Panel) Wire.result {
            var operation = E.beginDefault();
            defer operation.end();
            return chebyshevSeriesSetAdmitted(T, &operation, a, x, coefficients, counts, y);
        }
        pub fn chebyshevSeriesSetAgreedAdmitted(comptime T: type, operation: *const C.Operation, a: Action, agreement: Agreement, x: Wire.const_buffer, coefficients: ConstPanel, counts: Wire.const_buffer, y: Panel) Wire.result {
            return seriesSetEntryAdmitted(T, operation, a, agreement, x, coefficients, counts, y);
        }
        pub fn chebyshevSeriesSetAgreed(comptime T: type, a: Action, agreement: Agreement, x: Wire.const_buffer, coefficients: ConstPanel, counts: Wire.const_buffer, y: Panel) Wire.result {
            var operation = E.beginDefault();
            defer operation.end();
            return chebyshevSeriesSetAgreedAdmitted(T, &operation, a, agreement, x, coefficients, counts, y);
        }
        fn LinearArguments(comptime T: type) type {
            return struct { rhs: []const T, initial: ?[]const T, output: []T, dimension: usize, iterations: usize, target: *LinearResult(T), inverse_action: ?Action, reduction: ?Reduction(T) };
        }
        fn decodeLinear(comptime T: type, a: Action, inverse_raw: ?*const anyopaque, reduction_raw: ?*const anyopaque, b: Wire.const_buffer, initial_mode: i32, initial_buffer: Wire.const_buffer, dimension: i64, iterations: i64, relative: S.Real(T), absolute: S.Real(T), system_alpha: T, system_shift: T, out: Wire.buffer, target_raw: ?*anyopaque) !LinearArguments(T) {
            try shape(T, a);
            if (!N.Reductions.finite(T, system_alpha) or !N.Reductions.finite(T, system_shift)) return error.NonFiniteInput;
            if (initial_mode != 0 and initial_mode != 1) return error.InvalidOptions;
            const target = try summaryTarget(LinearResult(T), target_raw);
            const target_range = try N.Storage.byteRange(@intFromPtr(target), @sizeOf(LinearResult(T)));
            const rhs = try Checked.readBuffer(T, b);
            const initial = if (initial_mode == 1) try Checked.readBuffer(T, initial_buffer) else null;
            const output_values = try Checked.writeBuffer(T, out);
            const output_range = try chainRange(output_values);
            try rejectOverlap(target_range, output_range);
            try rejectOverlap(target_range, try chainRange(rhs));
            if (initial) |values| try rejectOverlap(target_range, try chainRange(values));
            const n = try Checked.countElements(dimension);
            const budget = try Checked.countElements(iterations);
            // A run uses at most three A actions per iteration plus the initial
            // residual action. Every reported wire count must remain i64.
            if (iterations > @divTrunc(std.math.maxInt(i64) - 1, 3)) return error.SizeOverflow;
            if (a.rows != a.columns or rhs.len != @as(usize, @intCast(a.rows)) or output_values.len < rhs.len or rhs.len > n) return error.DimensionMismatch;
            if (initial) |values| if (values.len != rhs.len) return error.DimensionMismatch;
            if (!std.math.isFinite(relative) or relative < 0 or !std.math.isFinite(absolute) or absolute < 0) return error.InvalidOptions;
            const reduction = try decodedReduction(T, reduction_raw, &.{ output_range, target_range });
            if (reduction == null and n != rhs.len) return error.MissingGlobalReduction;
            const inverse_action = try decodedInverse(T, a, inverse_raw, &.{ output_range, target_range });
            return .{ .rhs = rhs, .initial = initial, .output = output_values, .dimension = n, .iterations = budget, .target = target, .inverse_action = inverse_action, .reduction = reduction };
        }
        fn bicgstabEntryAdmitted(comptime T: type, operation: *const C.Operation, a: Action, inverse_action: ?*const anyopaque, reduction: ?*const anyopaque, agreement: ?Agreement, b: Wire.const_buffer, initial_mode: i32, initial: Wire.const_buffer, dimension: i64, iterations: i64, relative: S.Real(T), absolute: S.Real(T), system_alpha: T, system_shift: T, out: Wire.buffer, summary: ?*anyopaque) Wire.result {
            _ = operation.borrowedContext();
            if (agreement) |g| if (g.agree == null) return failure(error.PhaseAgreementFailed, 0);
            var state = State(T){ .action = a, .agreement = agreement };
            const sync = state.synchronization();
            P.enter(T, sync, operation, .bicgstab) catch |err| return state.agreedFailure(err);
            const args = P.resolve(sync, operation, .{ .stage = .preflight }, decodeLinear(T, a, inverse_action, reduction, b, initial_mode, initial, dimension, iterations, relative, absolute, system_alpha, system_shift, out, summary)) catch |err| return state.agreedFailure(err);
            state.inverse_action = args.inverse_action;
            state.reduction = args.reduction;
            const result = N.BiCGStab.Managed(E).runScratchAdmitted(T, operation, .{ .action = state.operator(), .action_scale = system_alpha, .diagonal_shift = system_shift, .preconditioner = state.inverseOperator(), .reduction = state.reductions(), .rhs = args.rhs, .initial = args.initial, .output = args.output, .global_dimension = args.dimension, .max_iterations = args.iterations, .relative_tolerance = relative, .absolute_tolerance = absolute }) catch |err| return state.agreedFailure(err);
            args.target.* = .{ .status = @intCast(@backingInt(result.status)), .reserved = 0, .iterations = @intCast(result.iterations), .action_count = @intCast(result.action_count), .preconditioner_count = @intCast(result.preconditioner_count), .residual_restarts = @intCast(result.residual_restarts), .residual_norm = result.residual_norm };
            return okay();
        }
        pub fn bicgstabAdmitted(comptime T: type, op: *const C.Operation, a: Action, inverse: ?*const anyopaque, reduction: ?*const anyopaque, b: Wire.const_buffer, initial_mode: i32, initial: Wire.const_buffer, n: i64, iterations: i64, relative: S.Real(T), absolute: S.Real(T), system_alpha: T, system_shift: T, out: Wire.buffer, summary: ?*anyopaque) Wire.result {
            return bicgstabEntryAdmitted(T, op, a, inverse, reduction, null, b, initial_mode, initial, n, iterations, relative, absolute, system_alpha, system_shift, out, summary);
        }
        pub fn bicgstab(comptime T: type, a: Action, inverse: ?*const anyopaque, reduction: ?*const anyopaque, b: Wire.const_buffer, initial_mode: i32, initial: Wire.const_buffer, n: i64, iterations: i64, relative: S.Real(T), absolute: S.Real(T), system_alpha: T, system_shift: T, out: Wire.buffer, summary: ?*anyopaque) Wire.result {
            var op = E.beginDefault();
            defer op.end();
            return bicgstabAdmitted(T, &op, a, inverse, reduction, b, initial_mode, initial, n, iterations, relative, absolute, system_alpha, system_shift, out, summary);
        }
        pub fn bicgstabAgreedAdmitted(comptime T: type, op: *const C.Operation, a: Action, inverse: ?*const anyopaque, reduction: ?*const anyopaque, agreement: Agreement, b: Wire.const_buffer, initial_mode: i32, initial: Wire.const_buffer, n: i64, iterations: i64, relative: S.Real(T), absolute: S.Real(T), system_alpha: T, system_shift: T, out: Wire.buffer, summary: ?*anyopaque) Wire.result {
            return bicgstabEntryAdmitted(T, op, a, inverse, reduction, agreement, b, initial_mode, initial, n, iterations, relative, absolute, system_alpha, system_shift, out, summary);
        }
        pub fn bicgstabAgreed(comptime T: type, a: Action, inverse: ?*const anyopaque, reduction: ?*const anyopaque, agreement: Agreement, b: Wire.const_buffer, initial_mode: i32, initial: Wire.const_buffer, n: i64, iterations: i64, relative: S.Real(T), absolute: S.Real(T), system_alpha: T, system_shift: T, out: Wire.buffer, summary: ?*anyopaque) Wire.result {
            var op = E.beginDefault();
            defer op.end();
            return bicgstabAgreedAdmitted(T, &op, a, inverse, reduction, agreement, b, initial_mode, initial, n, iterations, relative, absolute, system_alpha, system_shift, out, summary);
        }
        pub fn RationalResult(comptime T: type) type {
            return extern struct { status: i32, linear_status: i32, completed_systems: i64, stopped_node: i64, stopped_column: i64, iterations: i64, action_count: i64, preconditioner_count: i64, residual_restarts: i64, max_residual_norm: S.Real(T) };
        }
        fn decodedInverse(comptime T: type, a: Action, raw: ?*const anyopaque, outputs: []const N.Storage.ByteRange) !?Action {
            const pointer = raw orelse return null;
            if (@intFromPtr(pointer) % @alignOf(Action) != 0) return error.MisalignedPointer;
            const range = try N.Storage.byteRange(@intFromPtr(pointer), @sizeOf(Action));
            for (outputs) |out| try rejectOverlap(range, out);
            const descriptor: *const Action = @ptrCast(@alignCast(pointer));
            try shape(T, descriptor.*);
            if (descriptor.rows != a.rows or descriptor.columns != a.columns) return error.DimensionMismatch;
            return descriptor.*;
        }
        fn RationalArguments(comptime T: type) type {
            return struct { input: N.Dense.ConstPanel(T), nodes: []const T, weights: []const T, output: N.Dense.Panel(T), dimension: usize, iterations: usize, target: *RationalResult(T), inverse: ?Action, reduction: ?Reduction(T) };
        }
        fn decodeRational(comptime T: type, a: Action, inverse_raw: ?*const anyopaque, reduction_raw: ?*const anyopaque, x: ConstPanel, nodes_buffer: Wire.const_buffer, weights_buffer: Wire.const_buffer, dimension: i64, iterations: i64, relative: S.Real(T), absolute: S.Real(T), y: Panel, summary: ?*anyopaque) !RationalArguments(T) {
            try shape(T, a);
            const target = try summaryTarget(RationalResult(T), summary);
            const target_range = try N.Storage.byteRange(@intFromPtr(target), @sizeOf(RationalResult(T)));
            const input_panel: N.Dense.ConstPanel(T) = .{ .values = try Checked.readBuffer(T, x.dense.storage), .shape = try panelShape(x.dense, x.layout, x.reserved) };
            const output_panel: N.Dense.Panel(T) = .{ .values = try Checked.writeBuffer(T, y.dense.storage), .shape = try panelShape(y.dense, y.layout, y.reserved) };
            _ = try input_panel.check();
            _ = try output_panel.check();
            const nodes = try Checked.readBuffer(T, nodes_buffer);
            const weights = try Checked.readBuffer(T, weights_buffer);
            const n = try Checked.countElements(dimension);
            const budget = try Checked.countElements(iterations);
            if (a.rows != a.columns or input_panel.shape.row_count != @as(usize, @intCast(a.rows)) or input_panel.shape.row_count > n or output_panel.shape.row_count != input_panel.shape.row_count or output_panel.shape.column_count != input_panel.shape.column_count or nodes.len != weights.len) return error.DimensionMismatch;
            if (!std.math.isFinite(relative) or relative < 0 or !std.math.isFinite(absolute) or absolute < 0) return error.InvalidOptions;
            // Prove every aggregate counter fits the wire before publication.
            // The conservative bound includes zero-weight nodes and true-residual
            // checks; it is independent of numerical convergence and MPI rank.
            const limit = std.math.maxInt(i64);
            if (iterations > @divTrunc(limit - 1, 3)) return error.SizeOverflow;
            const systems = std.math.mul(u64, @intCast(nodes.len), @intCast(input_panel.shape.column_count)) catch return error.SizeOverflow;
            const actions = std.math.mul(u64, systems, @as(u64, @intCast(iterations)) * 3 + 1) catch return error.SizeOverflow;
            if (actions > limit) return error.SizeOverflow;
            const output_range = try chainRange(output_panel.values);
            for ([_]N.Storage.ByteRange{ try chainRange(input_panel.values), output_range, try chainRange(nodes), try chainRange(weights) }) |range| try rejectOverlap(target_range, range);
            const reduction = try decodedReduction(T, reduction_raw, &.{ output_range, target_range });
            if (reduction == null and n != input_panel.shape.row_count) return error.MissingGlobalReduction;
            const inverse = try decodedInverse(T, a, inverse_raw, &.{ output_range, target_range });
            return .{ .input = input_panel, .nodes = nodes, .weights = weights, .output = output_panel, .dimension = n, .iterations = budget, .target = target, .inverse = inverse, .reduction = reduction };
        }
        fn rationalFilterEntryAdmitted(comptime T: type, operation: *const C.Operation, a: Action, inverse: ?*const anyopaque, reduction: ?*const anyopaque, agreement: ?Agreement, x: ConstPanel, nodes: Wire.const_buffer, weights: Wire.const_buffer, dimension: i64, iterations: i64, relative: S.Real(T), absolute: S.Real(T), y: Panel, summary: ?*anyopaque) Wire.result {
            _ = operation.borrowedContext();
            if (agreement) |g| if (g.agree == null) return failure(error.PhaseAgreementFailed, 0);
            var state = State(T){ .action = a, .agreement = agreement };
            const sync = state.synchronization();
            P.enter(T, sync, operation, .rational_filter) catch |err| return state.agreedFailure(err);
            const args = P.resolve(sync, operation, .{ .stage = .preflight }, decodeRational(T, a, inverse, reduction, x, nodes, weights, dimension, iterations, relative, absolute, y, summary)) catch |err| return state.agreedFailure(err);
            state.inverse_action = args.inverse;
            state.reduction = args.reduction;
            const result = N.RationalFilter.Managed(E).runScratchAdmitted(T, operation, .{ .action = state.operator(), .preconditioner = state.inverseOperator(), .reduction = state.reductions(), .input = args.input, .nodes = args.nodes, .weights = args.weights, .output = args.output, .global_dimension = args.dimension, .max_iterations = args.iterations, .relative_tolerance = relative, .absolute_tolerance = absolute }) catch |err| return state.agreedFailure(err);
            args.target.* = .{ .status = @intCast(@backingInt(result.status)), .linear_status = if (result.linear_status) |v| @intCast(@backingInt(v)) else -1, .completed_systems = @intCast(result.completed_systems), .stopped_node = if (result.stopped_node) |v| @intCast(v) else -1, .stopped_column = if (result.stopped_column) |v| @intCast(v) else -1, .iterations = @intCast(result.iterations), .action_count = @intCast(result.action_count), .preconditioner_count = @intCast(result.preconditioner_count), .residual_restarts = @intCast(result.residual_restarts), .max_residual_norm = result.max_residual_norm };
            return okay();
        }
        pub fn rationalFilterAdmitted(comptime T: type, op: *const C.Operation, a: Action, inverse: ?*const anyopaque, reduction: ?*const anyopaque, x: ConstPanel, nodes: Wire.const_buffer, weights: Wire.const_buffer, n: i64, iterations: i64, relative: S.Real(T), absolute: S.Real(T), y: Panel, summary: ?*anyopaque) Wire.result {
            return rationalFilterEntryAdmitted(T, op, a, inverse, reduction, null, x, nodes, weights, n, iterations, relative, absolute, y, summary);
        }
        pub fn rationalFilter(comptime T: type, a: Action, inverse: ?*const anyopaque, reduction: ?*const anyopaque, x: ConstPanel, nodes: Wire.const_buffer, weights: Wire.const_buffer, n: i64, iterations: i64, relative: S.Real(T), absolute: S.Real(T), y: Panel, summary: ?*anyopaque) Wire.result {
            var op = E.beginDefault();
            defer op.end();
            return rationalFilterAdmitted(T, &op, a, inverse, reduction, x, nodes, weights, n, iterations, relative, absolute, y, summary);
        }
        pub fn rationalFilterAgreedAdmitted(comptime T: type, op: *const C.Operation, a: Action, inverse: ?*const anyopaque, reduction: ?*const anyopaque, agreement: Agreement, x: ConstPanel, nodes: Wire.const_buffer, weights: Wire.const_buffer, n: i64, iterations: i64, relative: S.Real(T), absolute: S.Real(T), y: Panel, summary: ?*anyopaque) Wire.result {
            return rationalFilterEntryAdmitted(T, op, a, inverse, reduction, agreement, x, nodes, weights, n, iterations, relative, absolute, y, summary);
        }
        pub fn rationalFilterAgreed(comptime T: type, a: Action, inverse: ?*const anyopaque, reduction: ?*const anyopaque, agreement: Agreement, x: ConstPanel, nodes: Wire.const_buffer, weights: Wire.const_buffer, n: i64, iterations: i64, relative: S.Real(T), absolute: S.Real(T), y: Panel, summary: ?*anyopaque) Wire.result {
            var op = E.beginDefault();
            defer op.end();
            return rationalFilterAgreedAdmitted(T, &op, a, inverse, reduction, agreement, x, nodes, weights, n, iterations, relative, absolute, y, summary);
        }
        pub fn ContourSummary(comptime T: type, comptime CW: type) type {
            return CW.Summary(S.Real(T), RationalResult(T));
        }
        fn ContourArguments(comptime T: type, comptime CW: type) type {
            return struct {
                options: CW.Options(S.Real(T)),
                input: N.Dense.ConstPanel(T),
                vectors: N.Dense.Panel(T),
                eigenvalues: []S.Real(T),
                residuals: []S.Real(T),
                target: *ContourSummary(T, CW),
                inverse: ?Action,
                reduction: ?Reduction(T),
                projection: ?BlockReduction,
                block: usize,
                real_count: usize,
            };
        }
        fn decodeContour(comptime T: type, comptime CW: type, a: Action, inverse_raw: ?*const anyopaque, reduction_raw: ?*const anyopaque, projection_raw: ?*const anyopaque, options_raw: ?*const anyopaque, x: ConstPanel, y: Panel, eigenvalues: Wire.buffer, residuals: Wire.buffer, summary: ?*anyopaque) !ContourArguments(T, CW) {
            try shape(T, a);
            const target = try summaryTarget(ContourSummary(T, CW), summary);
            const raw = options_raw orelse return error.NullPointer;
            const Options = CW.Options(S.Real(T));
            if (@intFromPtr(raw) % @alignOf(Options) != 0) return error.MisalignedPointer;
            const options_range = try N.Storage.byteRange(@intFromPtr(raw), @sizeOf(Options));
            const pointer: *const Options = @ptrCast(@alignCast(raw));
            const o = pointer.*;
            try CW.validateDescriptor(S.Real(T), o);
            const input_panel: N.Dense.ConstPanel(T) = .{ .values = try Checked.readBuffer(T, x.dense.storage), .shape = try panelShape(x.dense, x.layout, x.reserved) };
            const output_panel: N.Dense.Panel(T) = .{ .values = try Checked.writeBuffer(T, y.dense.storage), .shape = try panelShape(y.dense, y.layout, y.reserved) };
            _ = try input_panel.check();
            _ = try output_panel.check();
            const values = try Checked.writeBuffer(S.Real(T), eigenvalues);
            const errors = try Checked.writeBuffer(S.Real(T), residuals);
            const n = try Checked.countElements(a.rows);
            const dimension = try Checked.countElements(o.global_dimension);
            const m = input_panel.shape.column_count;
            if (a.rows != a.columns or n > dimension or m > dimension or
                input_panel.shape.row_count != n or output_panel.shape.row_count != n or
                output_panel.shape.column_count != m or (m == 0 and dimension != 0)) return error.DimensionMismatch;
            if (values.len < m or errors.len < m) return error.BufferTooSmall;
            const target_range = try N.Storage.byteRange(@intFromPtr(target), @sizeOf(ContourSummary(T, CW)));
            const ranges = [_]N.Storage.ByteRange{ try chainRange(output_panel.values), try chainRange(values), try chainRange(errors), target_range };
            for (ranges, 0..) |range, i| {
                try rejectOverlap(options_range, range);
                for (ranges[i + 1 ..]) |other| try rejectOverlap(range, other);
            }
            try rejectOverlap(target_range, try chainRange(input_panel.values));
            const inverse = try decodedInverse(T, a, inverse_raw, &ranges);
            const reduction = try decodedReduction(T, reduction_raw, &ranges);
            var projection: ?BlockReduction = null;
            if (projection_raw) |p| {
                if (@intFromPtr(p) % @alignOf(BlockReduction) != 0) return error.MisalignedPointer;
                const range = try N.Storage.byteRange(@intFromPtr(p), @sizeOf(BlockReduction));
                for (ranges) |out| try rejectOverlap(range, out);
                const descriptor: *const BlockReduction = @ptrCast(@alignCast(p));
                if (descriptor.reduce == null) return error.NullPointer;
                projection = descriptor.*;
            }
            return .{ .options = o, .input = input_panel, .vectors = output_panel, .eigenvalues = values, .residuals = errors, .target = target, .inverse = inverse, .reduction = reduction, .projection = projection, .block = std.math.mul(usize, n, m) catch return error.SizeOverflow, .real_count = std.math.mul(usize, m, 2) catch return error.SizeOverflow };
        }
        fn contourRequest(comptime T: type, comptime CW: type, state: *State(T), args: ContourArguments(T, CW)) N.ContourEigensolver.Request(T) {
            const o = args.options;
            return .{ .action = state.contourOperator(), .preconditioner = state.inverseOperator(), .reduction = state.reductions(), .projection_reduction = state.blockReducer(), .initial = args.input, .vectors = args.vectors, .eigenvalues = args.eigenvalues, .residuals = args.residuals, .global_dimension = @intCast(o.global_dimension), .contour = .{ .lower = o.lower, .upper = o.upper, .vertical_ratio = o.vertical_ratio, .half_nodes = @intCast(o.half_nodes), .rule = if (o.quadrature_rule == 0) .midpoint_trapezoidal else .gauss_legendre, .gauss_max_iterations = @intCast(o.gauss_max_iterations) }, .max_iterations = @intCast(o.max_iterations), .relative_tolerance = o.relative_tolerance, .absolute_tolerance = o.absolute_tolerance, .linear_max_iterations = @intCast(o.linear_max_iterations), .linear_relative_tolerance = o.linear_relative_tolerance, .linear_absolute_tolerance = o.linear_absolute_tolerance, .linear_accuracy = if (o.linear_accuracy == 0) .fixed else .residual_scaled, .allow_inexact_iteration_limit = o.flags & CW.allow_inexact_iteration_limit != 0, .rank_relative_tolerance = if (o.flags & CW.rank_relative_present != 0) o.rank_relative_tolerance else null, .rank_absolute_tolerance = o.rank_absolute_tolerance, .eigen_options = .{ .max_sweeps = @intCast(o.ritz_max_sweeps), .relative_tolerance = if (o.flags & CW.ritz_relative_present != 0) o.ritz_relative_tolerance else null }, .output_selection = if (o.output_selection == 0) .selected else .all_ritz, .publish_finite_stops = o.flags & CW.publish_finite_stops != 0, .convergence = if (o.convergence == 0) .residual else .trace, .trace_tolerance = o.trace_tolerance, .method = if (o.method == 0) .filtered_subspace else .residual_inverse };
        }
        fn contourCount(value: usize) !i64 {
            return std.math.cast(i64, value) orelse error.SizeOverflow;
        }
        fn encodeContour(comptime T: type, comptime CW: type, result: N.ContourEigensolver.Result(T)) !ContourSummary(T, CW) {
            var out = std.mem.zeroes(ContourSummary(T, CW));
            out.status = switch (result.status) {
                .converged => 0,
                .iteration_limit => 1,
                .capacity_limit => 2,
                .empty_subspace => 3,
                .linear_stop => 4,
                .ritz_stop => 5,
            };
            inline for (.{ "iterations", "rank", "found", "converged_pairs", "published", "completed_systems", "linear_iterations", "inexact_systems", "action_count" }) |field|
                @field(out, field) = try contourCount(@field(result, field));
            out.max_residual_norm = result.max_residual_norm;
            out.trace_change = result.trace_change;
            if (result.linear) |linear| {
                out.has_linear = 1;
                out.linear.status = @intCast(@backingInt(linear.status));
                out.linear.linear_status = if (linear.linear_status) |v| @intCast(@backingInt(v)) else -1;
                out.linear.stopped_node = if (linear.stopped_node) |v| try contourCount(v) else -1;
                out.linear.stopped_column = if (linear.stopped_column) |v| try contourCount(v) else -1;
                inline for (.{ "completed_systems", "iterations", "action_count", "preconditioner_count", "residual_restarts" }) |field|
                    @field(out.linear, field) = try contourCount(@field(linear, field));
                out.linear.max_residual_norm = linear.max_residual_norm;
            }
            if (result.ritz) |ritz| {
                out.has_ritz = 1;
                out.ritz.status = switch (ritz.status) {
                    .converged => 0,
                    .gram_iteration_limit => 1,
                    .reduced_iteration_limit => 2,
                    .indefinite_metric => 3,
                    .range_limit => 4,
                };
                out.ritz.rank = try contourCount(ritz.rank);
                out.ritz.gram_sweeps = try contourCount(ritz.gram_sweeps);
                out.ritz.reduced_sweeps = try contourCount(ritz.reduced_sweeps);
            }
            return out;
        }
        fn contourEntryAdmitted(comptime T: type, comptime CW: type, operation: *const C.Operation, a: Action, inverse: ?*const anyopaque, reduction: ?*const anyopaque, projection: ?*const anyopaque, agreement: ?Agreement, options_raw: ?*const anyopaque, x: ConstPanel, y: Panel, eigenvalues: Wire.buffer, residuals: Wire.buffer, summary: ?*anyopaque) Wire.result {
            _ = operation.borrowedContext();
            if (agreement) |g| if (g.agree == null) return failure(error.PhaseAgreementFailed, 0);
            var state = State(T){ .action = a, .agreement = agreement };
            const sync = state.contourSynchronization();
            P.enter(T, sync, operation, .contour_eigensolver) catch |err| return state.agreedFailure(contourError(err));
            const args = P.resolve(sync, operation, .{ .stage = .preflight }, decodeContour(T, CW, a, inverse, reduction, projection, options_raw, x, y, eigenvalues, residuals, summary)) catch |err| return state.agreedFailure(contourError(err));
            state.inverse_action = args.inverse;
            state.reduction = args.reduction;
            state.block_reduction = args.projection;
            var request = contourRequest(T, CW, &state, args);
            P.resolve(sync, operation, .{ .stage = .preflight }, N.ContourEigensolver.validateRequest(T, request)) catch |err| return state.agreedFailure(contourError(err));
            var vector_stage = P.acquire(C, T, sync, operation, args.block, .vector_allocation) catch |err| return state.agreedFailure(contourError(err));
            defer vector_stage.deinit();
            var real_stage = P.acquire(C, S.Real(T), sync, operation, args.real_count, .real_allocation) catch |err| return state.agreedFailure(contourError(err));
            defer real_stage.deinit();
            const m = args.input.shape.column_count;
            request.vectors = .{ .values = vector_stage.lease.values, .shape = .{ .row_count = args.input.shape.row_count, .column_count = m, .leading_dimension = args.input.shape.row_count, .layout = .column_major } };
            request.eigenvalues = real_stage.lease.values[0..m];
            request.residuals = real_stage.lease.values[m..];
            const result = N.ContourEigensolver.Managed(E).runScratchAdmitted(T, operation, request) catch |err| return state.agreedFailure(contourError(err));
            const encoded = P.resolve(sync, operation, .{ .stage = .commit, .index = 1 }, encodeContour(T, CW, result)) catch |err| return state.agreedFailure(contourError(err));
            P.check(sync, operation, .{ .stage = .commit, .index = 2, .control = @intCast(result.published) }) catch |err| return state.agreedFailure(contourError(err));
            for (0..result.published) |column| {
                args.eigenvalues[column] = request.eigenvalues[column];
                args.residuals[column] = request.residuals[column];
                for (0..args.input.shape.row_count) |row| args.vectors.column(column).at(row).* = request.vectors.column(column).at(row).*;
            }
            args.target.* = encoded;
            return okay();
        }
        pub fn contourAdmitted(comptime T: type, comptime CW: type, operation: *const C.Operation, a: Action, inverse: ?*const anyopaque, reduction: ?*const anyopaque, projection: ?*const anyopaque, options_raw: ?*const anyopaque, x: ConstPanel, y: Panel, eigenvalues: Wire.buffer, residuals: Wire.buffer, summary: ?*anyopaque) Wire.result {
            return contourEntryAdmitted(T, CW, operation, a, inverse, reduction, projection, null, options_raw, x, y, eigenvalues, residuals, summary);
        }
        pub fn contourAgreedAdmitted(comptime T: type, comptime CW: type, operation: *const C.Operation, a: Action, inverse: ?*const anyopaque, reduction: ?*const anyopaque, projection: ?*const anyopaque, agreement: Agreement, options_raw: ?*const anyopaque, x: ConstPanel, y: Panel, eigenvalues: Wire.buffer, residuals: Wire.buffer, summary: ?*anyopaque) Wire.result {
            return contourEntryAdmitted(T, CW, operation, a, inverse, reduction, projection, agreement, options_raw, x, y, eigenvalues, residuals, summary);
        }
        pub fn contour(comptime T: type, comptime CW: type, a: Action, inverse: ?*const anyopaque, reduction: ?*const anyopaque, projection: ?*const anyopaque, options_raw: ?*const anyopaque, x: ConstPanel, y: Panel, eigenvalues: Wire.buffer, residuals: Wire.buffer, summary: ?*anyopaque) Wire.result {
            var operation = E.beginDefault();
            defer operation.end();
            return contourAdmitted(T, CW, &operation, a, inverse, reduction, projection, options_raw, x, y, eigenvalues, residuals, summary);
        }
        pub fn contourAgreed(comptime T: type, comptime CW: type, a: Action, inverse: ?*const anyopaque, reduction: ?*const anyopaque, projection: ?*const anyopaque, agreement: Agreement, options_raw: ?*const anyopaque, x: ConstPanel, y: Panel, eigenvalues: Wire.buffer, residuals: Wire.buffer, summary: ?*anyopaque) Wire.result {
            var operation = E.beginDefault();
            defer operation.end();
            return contourAgreedAdmitted(T, CW, &operation, a, inverse, reduction, projection, agreement, options_raw, x, y, eigenvalues, residuals, summary);
        }
        pub fn fftwExecuteAdmitted(comptime FFT: type, token: ?*const Token, handle: ?*anyopaque) Wire.result {
            comptime {
                if (FFT.Provider.OwnerIdentity != Core) @compileError("operator FFT forwarding requires the exact shared Core");
            }
            const operation = borrowed(token) catch |err| return failure(err, 0);
            const plan = handle orelse return failure(error.NullPointer, 0);
            FFT.executeAdmitted(@ptrCast(plan), operation) catch |err| return failure(err, 0);
            return okay();
        }
        pub fn spmvAdmitted(comptime T: type, comptime O: type, comptime I: type, token: ?*const Token, a: Wire.csr, flag: i32, alpha: T, x: Wire.const_buffer, beta: T, y: Wire.buffer) Wire.result {
            const operation = borrowed(token) catch |err| return failure(err, 0);
            return Checked.spmvWithIndexTypesAdmitted(operation, T, O, I, a, flag, alpha, x, beta, y);
        }
        pub fn spmmAdmitted(comptime T: type, comptime O: type, comptime I: type, token: ?*const Token, a: Wire.csr, flag: i32, alpha: T, x: Wire.const_dense, beta: T, y: Wire.dense) Wire.result {
            const operation = borrowed(token) catch |err| return failure(err, 0);
            return Checked.spmmWithIndexTypesAdmitted(operation, T, O, I, a, flag, alpha, x, beta, y);
        }
    };
}
