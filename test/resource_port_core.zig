// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//! Explicit pure Zig fixture only; not a production provider or qualification.
const std = @import("std");
const E = @import("zynum-execution");
const core = @import("zynum-core");
const native = core.native;
const dense = core.internal.numerics.Dense;
const ports = E.resource_port;
const Fixture = struct {
    allocator: std.mem.Allocator,
    live: usize = 0,
    allocations: usize = 0,
    runs: usize = 0,
    releases: usize = 0,
    reject: bool = false,
    scratch_error: ?E.ExecutionError = null,
    scratch_backing: ?[]u8 = null,
    context: ?*E.Context = null,
    const Lease = struct {
        owner: *Fixture,
        pointer: [*]u8,
        bytes: usize,
        alignment: std.mem.Alignment,
        owned: bool,
    };
    fn cast(raw: *anyopaque) *Fixture {
        return @ptrCast(@alignCast(raw));
    }
    fn acquire(raw: *anyopaque, bytes: usize, alignment: usize, report: *E.CallReport) E.ExecutionError!ports.ByteLease {
        const self = cast(raw);
        if (self.context.?.snapshot().resource_pins == 0) @panic("allocation before pin");
        self.allocations += 1;
        report.provider.stage = .scratch_allocate;
        if (self.scratch_error) |err| return err;
        const a = std.mem.Alignment.fromByteUnits(alignment);
        const pointer = if (self.scratch_backing) |backing| available: {
            if (backing.len < @max(bytes, 1) or @intFromPtr(backing.ptr) % alignment != 0) return error.InvalidProvider;
            break :available backing.ptr;
        } else self.allocator.rawAlloc(@max(bytes, 1), a, @returnAddress()) orelse return error.OutOfMemory;
        self.live += 1;
        return .{ .values = pointer[0..bytes], .alignment = alignment, .ownership = ports.Ownership.init(Lease, .{ .owner = self, .pointer = pointer, .bytes = @max(bytes, 1), .alignment = a, .owned = self.scratch_backing == null }), .release_fn = free };
    }
    fn free(storage: *const ports.Ownership, _: *E.CallReport) void {
        const lease = storage.value(Lease);
        if (lease.owner.context.?.snapshot().resource_pins == 0) @panic("release after pin");
        if (lease.owned) lease.owner.allocator.rawFree(lease.pointer[0..lease.bytes], lease.alignment, @returnAddress());
        lease.owner.live -= 1;
    }
    fn run(raw: *anyopaque, batch: *const ports.TaskBatch, report: *E.CallReport) E.ExecutionError!void {
        const self = cast(raw);
        if (self.context.?.snapshot().active_operations != 1) @panic("expected one admission");
        self.runs += 1;
        report.provider.stage = .submit;
        if (self.reject) {
            report.provider.returned_status = -27181; // Opaque to Core; no external status namespace.
            report.provider.state = .rejected;
            return error.ProviderRejected;
        }
        for (0..batch.count) |i| batch.task_fn(batch.tasks, i);
        report.provider.state = .quiescent;
        report.provider.callbacks_started = batch.count;
        report.provider.callbacks_completed = batch.count;
        report.output = .complete;
    }
    fn release(raw: *anyopaque, _: *E.CallReport) bool {
        const self = cast(raw);
        const snapshot = self.context.?.snapshot();
        if (snapshot.active_operations != 0 or snapshot.resource_pins != 0 or snapshot.phase != .releasing or self.live != 0) @panic("invalid release ordering");
        self.releases += 1;
        return true;
    }
    const vtable: ports.VTable = .{ .acquire_bytes = acquire, .run_sync = run, .release_provider = release };
    fn port(self: *Fixture) ports.ResourcePort {
        return .{ .owner = self, .vtable = &vtable, .parallel_limit = 1 };
    }
};
test "header-free single Core uses explicit resource port and preserves rejection" {
    var fixture: Fixture = .{ .allocator = std.testing.allocator };
    var context = E.Context.initResources(fixture.port());
    fixture.context = &context;
    var report: E.CallReport = .{};
    const a = [_]f64{ 1, 2, 3, 4, 5, 6 };
    const b = [_]f64{ 7, 8, 9, 10, 11, 12 };
    var c = [_]f64{ 0, 0, 0, 0 };
    const args: native.MatrixProductF64 = .{
        .left_matrix = .{ .values = &a, .row_count = 2, .column_count = 3, .leading_dimension = 2 },
        .right_matrix = .{ .values = &b, .row_count = 3, .column_count = 2, .leading_dimension = 3 },
        .result_matrix = .{ .values = &c, .row_count = 2, .column_count = 2, .leading_dimension = 2 },
    };
    try native.matrixMultiplyF64NN(&context, args, &report);
    try std.testing.expectEqualSlices(f64, &.{ 76, 100, 103, 136 }, &c);
    fixture.reject = true;
    try std.testing.expectError(error.ProviderRejected, native.matrixMultiplyF64NN(&context, args, &report));
    try std.testing.expectEqual(@as(i32, -27181), report.provider.returned_status);
    try std.testing.expectEqual(E.OutputState.unchanged, report.output);
    try std.testing.expectEqualSlices(f64, &.{ 76, 100, 103, 136 }, &c);
    try std.testing.expectEqual(@as(usize, 0), fixture.live);
    try std.testing.expectEqual(@as(usize, 2), fixture.runs);
    _ = context.requestClose();
    try std.testing.expectEqual(E.CloseResult.closed, context.tryProgressClose(&report));
    try std.testing.expectEqual(@as(usize, 1), fixture.releases);
}
test "neutral scratch keeps exact ownership and pin until released after operation" {
    var fixture: Fixture = .{ .allocator = std.testing.allocator };
    var context = E.Context.initResources(fixture.port());
    fixture.context = &context;
    var report: E.CallReport = .{};
    var operation = try context.begin();
    const T = struct { byte: u8 align(64) };
    var scratch = try E.acquireOperationScratch(&operation, T, 0, &report);
    try std.testing.expectEqual(@as(usize, 0), scratch.values.len);
    try std.testing.expectEqual(@as(usize, 0), @intFromPtr(scratch.values.ptr) % 64);
    try std.testing.expectError(error.ExtentOverflow, E.acquireOperationScratch(&operation, f64, std.math.maxInt(usize), &report));
    try std.testing.expectEqual(@as(usize, 1), fixture.allocations);
    operation.end();
    _ = context.requestClose();
    try std.testing.expectEqual(E.CloseResult.waiting_resources, context.tryProgressClose(&report));
    scratch.release(&report);
    try std.testing.expectEqual(E.CloseResult.closed, context.tryProgressClose(&report));
}

const G = core.internal.lapack_checked.generalized_symmetric_eigen;
const ritz_h = [_]f64{ 4.8, -2.4, -2.4, -3.8 };
const ritz_s = [_]f64{ 4, 2, 2, 2 };

test "real generalized eigenpairs use one neutral lease and preserve strided padding" {
    var fixture: Fixture = .{ .allocator = std.testing.allocator };
    var context = E.Context.initResources(fixture.port());
    fixture.context = &context;
    var report: E.CallReport = .{};
    var operation = try context.begin();
    defer {
        operation.end();
        _ = context.requestClose();
        if (context.tryProgressClose(&report) != .closed) @panic("failed to close test context");
    }

    const h = [_]f64{ 4.8, -2.4, std.math.nan(f64), -2.4, -3.8, 91 };
    const s = [_]f64{ 4, 2, 92, 2, 2, 93 };
    var values = [_]f64{ 101, 102, 103 };
    var vectors = [_]f64{ 104, 105, 106, 107, 108, 109 };
    const result = try G.solveAdmitted(&operation, .{
        .h = .{ .values = &h, .row_count = 2, .column_count = 2, .leading_dimension = 3 },
        .s = .{ .values = &s, .row_count = 2, .column_count = 2, .leading_dimension = 3 },
        .eigenvalues = &values,
        .coefficients = .{ .values = &vectors, .row_count = 2, .column_count = 2, .leading_dimension = 3 },
    }, &report);
    try std.testing.expectEqual(G.Status.converged, result.status);
    try std.testing.expectEqual(@as(usize, 0), result.failing_pivot);
    try std.testing.expectEqual(E.OutputState.complete, report.output);
    try std.testing.expectEqual(E.SubmissionState.not_submitted, report.provider.state);
    try std.testing.expectEqual(@as(usize, 1), fixture.allocations);
    try std.testing.expectEqual(@as(usize, 0), fixture.runs);
    try std.testing.expectEqual(@as(usize, 0), fixture.live);
    const snapshot = context.snapshot();
    try std.testing.expectEqual(@as(usize, 1), snapshot.active_operations);
    try std.testing.expectEqual(@as(usize, 0), snapshot.resource_pins);

    try std.testing.expectApproxEqAbs(@as(f64, -2), values[0], 1e-13);
    try std.testing.expectApproxEqAbs(@as(f64, 3), values[1], 1e-13);
    try std.testing.expectEqual(@as(f64, 103), values[2]);
    // C = L^-T Q, with L = [[2,0],[1,1]] and each column's sign free.
    const expected = [2][2]f64{ .{ -0.1, 0.8 }, .{ -0.7, 0.6 } };
    for (expected, 0..) |column, j| {
        const sign: f64 = if (vectors[j * 3 + 1] < 0) -1 else 1;
        for (column, 0..) |v, i| try std.testing.expectApproxEqAbs(v, sign * vectors[i + j * 3], 1e-13);
    }
    try std.testing.expectEqual(@as(f64, 106), vectors[2]);
    try std.testing.expectEqual(@as(f64, 109), vectors[5]);
}

test "real generalized eigensolve preserves outputs on pivot, arithmetic and convergence failures" {
    var fixture: Fixture = .{ .allocator = std.testing.allocator };
    var context = E.Context.initResources(fixture.port());
    fixture.context = &context;
    var report: E.CallReport = .{};
    var operation = try context.begin();
    defer {
        operation.end();
        _ = context.requestClose();
        if (context.tryProgressClose(&report) != .closed) @panic("failed to close test context");
    }

    const indefinite = [_]f64{ 4, 2, 2, 0.5 };
    const singular = [_]f64{ 2, 2, 2, 2 };
    const identity = [_]f64{ 1, 0, 0, 1 };
    const largest = std.math.floatMax(f64);
    const overflow_pivot = [_]f64{ 1, largest, largest, 1 };
    const overflow_factor = [_]f64{ 1e-308, 1e308, 1e308, 1 };
    const tiny = [_]f64{ 1e-300, 0, 0, 1e-300 };
    const huge_h = [_]f64{ largest, 0, 0, largest };
    const cases = [_]struct {
        h: []const f64 = &ritz_h,
        s: []const f64,
        sweeps: usize = 64,
        status: G.Status,
        pivot: usize = 0,
    }{
        .{ .s = &indefinite, .status = .not_positive_definite, .pivot = 2 },
        .{ .h = &identity, .s = &singular, .status = .accuracy_not_met },
        .{ .s = &overflow_pivot, .status = .nonfinite_arithmetic },
        .{ .s = &overflow_factor, .status = .nonfinite_arithmetic },
        .{ .h = &huge_h, .s = &tiny, .status = .nonfinite_arithmetic },
        .{ .s = &ritz_s, .sweeps = 0, .status = .iteration_limit },
    };
    var values = [_]f64{ 10, 11 };
    var vectors = [_]f64{ 12, 13, 14, 15 };
    for (cases) |case| {
        const result = try G.solveAdmitted(&operation, .{
            .h = .{ .values = case.h, .row_count = 2, .column_count = 2, .leading_dimension = 2 },
            .s = .{ .values = case.s, .row_count = 2, .column_count = 2, .leading_dimension = 2 },
            .eigenvalues = &values,
            .coefficients = .{ .values = &vectors, .row_count = 2, .column_count = 2, .leading_dimension = 2 },
            .options = .{ .max_sweeps = case.sweeps },
        }, &report);
        // A singular metric can also be rejected directly by its second
        // factorization pivot if the rounded Cholesky kernel detects it.
        if (case.status == .accuracy_not_met and result.status == .not_positive_definite) {
            try std.testing.expectEqual(@as(usize, 2), result.failing_pivot);
        } else {
            try std.testing.expectEqual(case.status, result.status);
            try std.testing.expectEqual(case.pivot, result.failing_pivot);
        }
        if (case.status != .accuracy_not_met) try std.testing.expectEqual(@as(usize, 0), result.sweeps);
        try std.testing.expectEqual(E.OutputState.unchanged, report.output);
        try std.testing.expectEqual(@as(usize, 0), fixture.live);
        try std.testing.expectEqual(@as(usize, 0), context.snapshot().resource_pins);
        try std.testing.expectEqualSlices(f64, &.{ 10, 11 }, &values);
        try std.testing.expectEqualSlices(f64, &.{ 12, 13, 14, 15 }, &vectors);
    }
    try std.testing.expectEqual(cases.len, fixture.allocations);
    try std.testing.expectEqual(@as(usize, 0), fixture.runs);
}

test "real generalized eigensolve rejects bad views before scratch and preserves outputs on budget rejection" {
    var fixture: Fixture = .{ .allocator = std.testing.allocator };
    var context = E.Context.initResources(fixture.port());
    fixture.context = &context;
    var report: E.CallReport = .{};
    var operation = try context.begin();
    defer {
        operation.end();
        _ = context.requestClose();
        if (context.tryProgressClose(&report) != .closed) @panic("failed to close test context");
    }

    var values = [_]f64{ 21, 22 };
    var vectors = [_]f64{ 23, 24, 25, 26, 27, 28 };
    const args: G.Arguments = .{
        .h = .{ .values = &ritz_h, .row_count = 2, .column_count = 2, .leading_dimension = 2 },
        .s = .{ .values = &ritz_s, .row_count = 2, .column_count = 2, .leading_dimension = 2 },
        .eigenvalues = &values,
        .coefficients = .{ .values = &vectors, .row_count = 2, .column_count = 2, .leading_dimension = 3 },
    };
    var bad = args;
    bad.coefficients.column_count = 1;
    try std.testing.expectError(error.DimensionMismatch, G.solveAdmitted(&operation, bad, &report));
    bad = args;
    bad.h.leading_dimension = 1;
    try std.testing.expectError(error.InvalidLeadingDimension, G.solveAdmitted(&operation, bad, &report));
    bad = args;
    bad.s.values = ritz_s[0..3];
    try std.testing.expectError(error.BufferTooSmall, G.solveAdmitted(&operation, bad, &report));
    bad = args;
    bad.h.leading_dimension = std.math.maxInt(usize);
    try std.testing.expectError(error.ExtentOverflow, G.solveAdmitted(&operation, bad, &report));
    bad = args;
    const too_wide = @as(usize, std.math.maxInt(i32)) + 1;
    bad.h.row_count = too_wide;
    bad.h.column_count = too_wide;
    bad.s.row_count = too_wide;
    bad.s.column_count = too_wide;
    bad.coefficients.row_count = too_wide;
    bad.coefficients.column_count = too_wide;
    try std.testing.expectError(error.ExtentOverflow, G.solveAdmitted(&operation, bad, &report));
    bad = args;
    bad.eigenvalues = vectors[2..4];
    try std.testing.expectError(error.AliasingNotAllowed, G.solveAdmitted(&operation, bad, &report));
    bad = args;
    bad.s.values = &.{ 4, 2, 3, 2 };
    try std.testing.expectError(error.NonSymmetricInput, G.solveAdmitted(&operation, bad, &report));
    bad = args;
    bad.h.values = &.{ 4, 0.0, -0.0, 2 };
    try std.testing.expectError(error.NonSymmetricInput, G.solveAdmitted(&operation, bad, &report));
    bad = args;
    bad.h.values = &.{ 4, 2, std.math.nan(f64), 2 };
    try std.testing.expectError(error.NonFiniteInput, G.solveAdmitted(&operation, bad, &report));
    try std.testing.expectEqual(@as(usize, 0), fixture.allocations);

    fixture.scratch_error = error.ResourceBudgetExceeded;
    try std.testing.expectError(error.ResourceBudgetExceeded, G.solveAdmitted(&operation, args, &report));
    try std.testing.expectEqual(E.ResourceStage.scratch_allocate, report.provider.stage);
    try std.testing.expectEqual(E.OutputState.unchanged, report.output);
    try std.testing.expectEqual(@as(usize, 0), context.snapshot().resource_pins);
    try std.testing.expectEqual(@as(usize, 1), fixture.allocations);
    try std.testing.expectEqual(@as(usize, 0), fixture.live);
    try std.testing.expectEqualSlices(f64, &.{ 21, 22 }, &values);
    try std.testing.expectEqualSlices(f64, &.{ 23, 24, 25, 26, 27, 28 }, &vectors);
}

test "real generalized eigensolve accepts shared read-only inputs and an empty typed problem" {
    var fixture: Fixture = .{ .allocator = std.testing.allocator };
    var context = E.Context.initResources(fixture.port());
    fixture.context = &context;
    var report: E.CallReport = .{};
    var operation = try context.begin();
    defer {
        operation.end();
        _ = context.requestClose();
        if (context.tryProgressClose(&report) != .closed) @panic("failed to close test context");
    }

    const empty: dense.Matrix(f64) = .{ .values = &.{}, .row_count = 0, .column_count = 0, .leading_dimension = 0 };
    const no_work = try G.solveAdmitted(&operation, .{
        .h = empty.asConst(),
        .s = empty.asConst(),
        .eigenvalues = &.{},
        .coefficients = empty,
    }, &report);
    try std.testing.expectEqual(G.Status.converged, no_work.status);
    try std.testing.expectEqual(E.OutputState.complete, report.output);
    try std.testing.expectEqual(@as(usize, 0), fixture.allocations);
    const source = [_]f64{4};
    var values = [_]f64{-1};
    var vectors = [_]f64{-1};
    const shared: dense.ConstMatrix(f64) = .{ .values = &source, .row_count = 1, .column_count = 1, .leading_dimension = 1 };
    const outcome = try G.solveAdmitted(&operation, .{
        .h = shared,
        .s = shared,
        .eigenvalues = &values,
        .coefficients = .{ .values = &vectors, .row_count = 1, .column_count = 1, .leading_dimension = 1 },
    }, &report);
    try std.testing.expectEqual(G.Status.converged, outcome.status);
    try std.testing.expectEqual(E.OutputState.complete, report.output);
    try std.testing.expectEqual(@as(f64, 1), values[0]);
    try std.testing.expectEqual(@as(f64, 0.5), vectors[0]);
    try std.testing.expectEqual(@as(usize, 1), fixture.allocations);

    // Extreme but full-rank S does not require a hidden rank cutoff. A scaled
    // vector is correct in the S metric even when its Euclidean norm is large.
    const extreme = [_]f64{ 1e-300, 0, 0, 1e300 };
    const diagonal: dense.ConstMatrix(f64) = .{ .values = &extreme, .row_count = 2, .column_count = 2, .leading_dimension = 2 };
    var extreme_values: [2]f64 = undefined;
    var extreme_vectors: [4]f64 = undefined;
    const scaled = try G.solveAdmitted(&operation, .{
        .h = diagonal,
        .s = diagonal,
        .eigenvalues = &extreme_values,
        .coefficients = .{ .values = &extreme_vectors, .row_count = 2, .column_count = 2, .leading_dimension = 2 },
    }, &report);
    try std.testing.expectEqual(G.Status.converged, scaled.status);
    for (extreme_values) |lambda| try std.testing.expectApproxEqAbs(@as(f64, 1), lambda, 1e-13);
    // A homogeneous zero H gives a zero residual and a zero residual scale.
    const zeros = [_]f64{0};
    const zero_h: dense.ConstMatrix(f64) = .{ .values = &zeros, .row_count = 1, .column_count = 1, .leading_dimension = 1 };
    const homogeneous = try G.solveAdmitted(&operation, .{
        .h = zero_h,
        .s = shared,
        .eigenvalues = &values,
        .coefficients = .{ .values = &vectors, .row_count = 1, .column_count = 1, .leading_dimension = 1 },
    }, &report);
    try std.testing.expectEqual(G.Status.converged, homogeneous.status);
    try std.testing.expectEqual(@as(f64, 0), values[0]);
    try std.testing.expectEqual(@as(f64, 0.5), vectors[0]);
    try std.testing.expectEqual(@as(usize, 3), fixture.allocations);
}

test "real generalized eigensolve refuses scratch across a backing tail or control span" {
    const untouched: [32]f64 = @splat(81);
    var vectors: [32]f64 align(16) = untouched;
    var values = [_]f64{ 91, 92 };
    const borrowed_scratch = std.mem.sliceAsBytes(vectors[8..26]);
    var fixture: Fixture = .{ .allocator = std.testing.allocator, .scratch_backing = borrowed_scratch };
    var context = E.Context.initResources(fixture.port());
    fixture.context = &context;
    var report: E.CallReport = .{};
    var operation = try context.begin();
    defer {
        operation.end();
        _ = context.requestClose();
        if (context.tryProgressClose(&report) != .closed) @panic("failed to close test context");
    }

    var args: G.Arguments = .{
        .h = .{ .values = &ritz_h, .row_count = 2, .column_count = 2, .leading_dimension = 2 },
        .s = .{ .values = &ritz_s, .row_count = 2, .column_count = 2, .leading_dimension = 2 },
        .eigenvalues = &values,
        .coefficients = .{ .values = &vectors, .row_count = 2, .column_count = 2, .leading_dimension = 2 },
    };
    try std.testing.expectError(error.InvalidProvider, G.solveAdmitted(&operation, args, &report));
    try std.testing.expectEqual(E.OutputState.unchanged, report.output);
    try std.testing.expectEqual(@as(usize, 0), fixture.live);
    try std.testing.expectEqualSlices(f64, &untouched, &vectors);
    try std.testing.expectEqualSlices(f64, &.{ 91, 92 }, &values);

    const control = try G.scratchExclusion(@intFromPtr(borrowed_scratch.ptr), borrowed_scratch.len);
    args.coefficients.values = vectors[0..4];
    args.excluded_scratch = &.{control};
    try std.testing.expectError(error.InvalidProvider, G.solveAdmitted(&operation, args, &report));
    try std.testing.expectEqual(E.OutputState.unchanged, report.output);
    try std.testing.expectEqual(@as(usize, 2), fixture.allocations);
    try std.testing.expectEqual(@as(usize, 0), fixture.live);
    try std.testing.expectEqual(@as(usize, 0), context.snapshot().resource_pins);
    try std.testing.expectEqualSlices(f64, &untouched, &vectors);
    try std.testing.expectEqualSlices(f64, &.{ 91, 92 }, &values);
}
