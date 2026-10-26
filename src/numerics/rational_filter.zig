// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Sum_j weight_j * (node_j I - A)^-1 B over borrowed Dense panels.
//! Explicit nodes/weights leave contour policy outside this neutral primitive.
//! All shifted systems use shared BiCGStab under the original operation. A
//! controlled linear stop reports its location and preserves the complete
//! output. An explicit inexact policy accepts a finite iteration-limited solve
//! and counts it separately. Widened sums narrow after all required solves finish.
//! No allocator policy, thread/handle manager or communicator owner is added.
const std = @import("std");
const S = @import("../blas/core/shared/scalar.zig");
const D = @import("dense.zig");
const O = @import("operator.zig");
const B = @import("bicgstab.zig");
const A = @import("recurrence_arithmetic.zig");
const Red = @import("reductions.zig");
const Store = @import("storage.zig");
const E = @import("zynum-execution");
const P = @import("synchronization.zig");

pub const Status = enum { success, linear_stop };
pub const LinearPolicy = enum { converged, finite_iteration_limit };
pub const Workspace = struct { elements: usize, wide_elements: usize };
/// Residual-inverse form: input is R=A*X-X*Lambda and basis is X. Each
/// column accumulates weight*(X+(node I-A)^-1 R)/(node-lambda).
/// The exact result equals the ordinary filter of X; approximate inner solves
/// now perturb a residual-sized RHS. All descriptors remain borrowed.
pub fn RitzCorrection(comptime T: type) type {
    return struct { basis: D.ConstPanel(T), eigenvalues: []const S.Real(T) };
}
pub fn Result(comptime T: type) type {
    return struct {
        status: Status = .success,
        completed_systems: usize = 0,
        inexact_systems: usize = 0,
        iterations: usize = 0,
        action_count: usize = 0,
        preconditioner_count: usize = 0,
        residual_restarts: usize = 0,
        stopped_node: ?usize = null,
        stopped_column: ?usize = null,
        linear_status: ?B.Status = null,
        max_residual_norm: S.Real(T) = 0,
    };
}
pub fn Request(comptime T: type) type {
    return struct {
        action: O.Operator(T),
        preconditioner: ?O.Operator(T) = null,
        reduction: ?B.Reduction(T) = null,
        input: D.ConstPanel(T),
        nodes: []const T,
        weights: []const T,
        output: D.Panel(T),
        global_dimension: usize,
        max_iterations: usize,
        relative_tolerance: S.Real(T),
        absolute_tolerance: S.Real(T) = 0,
        linear_policy: LinearPolicy = .converged,
        ritz_correction: ?RitzCorrection(T) = null,
    };
}
fn supported(comptime T: type) void {
    if (T != S.ComplexF32 and T != S.ComplexF64) @compileError("rational filter requires complex single or double precision");
}
pub fn Wide(comptime T: type) type {
    supported(T);
    return A.Wide(T);
}
fn product(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch return error.SizeOverflow;
}
fn sum(a: usize, b: usize) !usize {
    return std.math.add(usize, a, b) catch return error.SizeOverflow;
}
/// Empty columns, empty nodes and globally empty systems need no workspace.
/// For nonempty geometry the footprint is independent of numerical weights.
pub fn workspaceLength(comptime T: type, q: Request(T)) !Workspace {
    supported(T);
    const columns = q.input.shape.column_count;
    if (columns == 0 or q.nodes.len == 0 or q.global_dimension == 0) return .{ .elements = 0, .wide_elements = 0 };
    const panel = try product(q.action.rows, columns);
    const vectors = try sum(try product(panel, if (q.ritz_correction != null) 3 else 2), q.action.rows);
    const elements = try sum(vectors, try B.workspaceLength(T, q.action, q.preconditioner));
    const wide_elements = try product(panel, 2);
    _ = try Store.byteLength(elements, @sizeOf(T));
    _ = try Store.byteLength(wide_elements, @sizeOf(Wide(T)));
    return .{ .elements = elements, .wide_elements = wide_elements };
}
fn range(values: anytype) !Store.ByteRange {
    return Store.byteRange(@intFromPtr(values.ptr), try Store.byteLength(values.len, @sizeOf(@typeInfo(@TypeOf(values)).pointer.child)));
}
fn separate(a: anytype, b: anytype) !void {
    if (Store.overlaps(try range(a), try range(b))) return error.AliasingNotAllowed;
}
fn options(comptime T: type, q: Request(T)) !void {
    supported(T);
    const n = q.action.rows;
    const columns = q.input.shape.column_count;
    _ = try q.input.check();
    _ = try q.output.check();
    if (q.action.columns != n or q.input.shape.row_count != n or
        q.output.shape.row_count != n or q.output.shape.column_count != columns or
        q.nodes.len != q.weights.len or n > q.global_dimension) return error.DimensionMismatch;
    if (q.reduction == null and n != q.global_dimension) return error.MissingGlobalReduction;
    if (!std.math.isFinite(q.relative_tolerance) or q.relative_tolerance < 0 or
        !std.math.isFinite(q.absolute_tolerance) or q.absolute_tolerance < 0) return error.InvalidOptions;
    if (q.preconditioner) |inverse| {
        if (inverse.rows != n or inverse.columns != n) return error.DimensionMismatch;
        if (inverse.synchronization) |sync| {
            const parent = q.action.synchronization orelse return error.InvalidOptions;
            if (sync.context != parent.context or sync.agree_fn != parent.agree_fn) return error.InvalidOptions;
        }
    }
    const target = q.output.values[0..try q.output.check()];
    try separate(target, q.nodes);
    try separate(target, q.weights);
    var active = false;
    for (q.nodes, q.weights) |node, weight| {
        if (!Red.finite(T, node) or !Red.finite(T, weight)) return error.NonFiniteInput;
        active = active or !S.isZero(T, weight);
    }
    if (active and n != 0 and columns != 0) for (0..columns) |column| for (0..n) |row| {
        if (!Red.finite(T, q.input.column(column).at(row))) return error.NonFiniteInput;
    };
    if (q.ritz_correction) |correction| {
        _ = try correction.basis.check();
        if (correction.basis.shape.row_count != n or correction.basis.shape.column_count != columns or correction.eigenvalues.len < columns) return error.DimensionMismatch;
        try separate(target, correction.eigenvalues[0..columns]);
        for (correction.eigenvalues[0..columns]) |lambda| if (!std.math.isFinite(lambda)) return error.NonFiniteInput;
        if (active) for (0..columns) |column| {
            for (0..n) |row| if (!Red.finite(T, correction.basis.column(column).at(row))) return error.NonFiniteInput;
            for (q.nodes, q.weights) |node, weight| if (!S.isZero(T, weight) and node.re == correction.eigenvalues[column] and node.im == 0) return error.ShiftCoincidesWithRitzValue;
        };
    }
}
fn preflight(comptime T: type, q: Request(T), values: []T, wide: []Wide(T)) !Workspace {
    try options(T, q);
    const length = try workspaceLength(T, q);
    if (values.len < length.elements or wide.len < length.wide_elements) return error.WorkspaceTooSmall;
    const work = values[0..length.elements];
    const accum = wide[0..length.wide_elements];
    try separate(work, accum);
    const input = q.input.values[0..try q.input.check()];
    const output = q.output.values[0..try q.output.check()];
    for ([_][]const T{ input, output, q.nodes, q.weights }) |v| {
        try separate(v, work);
        try separate(v, accum);
    }
    if (q.ritz_correction) |correction| {
        const basis = correction.basis.values[0..try correction.basis.check()];
        try separate(basis, work);
        try separate(basis, accum);
        try separate(correction.eigenvalues[0..q.input.shape.column_count], work);
        try separate(correction.eigenvalues[0..q.input.shape.column_count], accum);
    }
    return length;
}
fn addCounts(comptime T: type, result: *Result(T), linear: B.Result(T)) !void {
    result.iterations = try sum(result.iterations, linear.iterations);
    result.action_count = try sum(result.action_count, linear.action_count);
    result.preconditioner_count = try sum(result.preconditioner_count, linear.preconditioner_count);
    result.residual_restarts = try sum(result.residual_restarts, linear.residual_restarts);
    result.max_residual_norm = @max(result.max_residual_norm, linear.residual_norm);
}
fn accumulate(comptime T: type, weights: T, values: []const T, real: []Wide(T), imaginary: []Wide(T)) !void {
    const W = Wide(T);
    for (values, real, imaginary) |x, *re, *im| {
        re.* += @as(W, weights.re) * x.re - @as(W, weights.im) * x.im;
        im.* += @as(W, weights.re) * x.im + @as(W, weights.im) * x.re;
        if (!std.math.isFinite(re.*) or !std.math.isFinite(im.*)) return error.NonFiniteResult;
    }
}
fn accumulateCorrection(comptime T: type, weight: T, node: T, lambda: S.Real(T), basis: []const T, solution: []const T, real: []Wide(T), imaginary: []Wide(T)) !void {
    const W = Wide(T);
    const dr = @as(W, node.re) - lambda;
    const di: W = node.im;
    const scale = @max(@abs(dr), @abs(di));
    if (scale == 0) return error.ShiftCoincidesWithRitzValue;
    const nr = dr / scale;
    const ni = di / scale;
    const denominator = nr * nr + ni * ni;
    const wr = @as(W, weight.re) / scale;
    const wi = @as(W, weight.im) / scale;
    const cr = (wr * nr + wi * ni) / denominator;
    const ci = (wi * nr - wr * ni) / denominator;
    for (basis, solution, real, imaginary) |x, y, *re, *im| {
        const xr = @as(W, x.re) + y.re;
        const xi = @as(W, x.im) + y.im;
        re.* += cr * xr - ci * xi;
        im.* += cr * xi + ci * xr;
        if (!std.math.isFinite(re.*) or !std.math.isFinite(im.*)) return error.NonFiniteResult;
    }
}
fn algorithm(comptime T: type, q: Request(T)) P.Algorithm {
    if (q.ritz_correction != null) return if (q.linear_policy == .converged) .residual_contour_filter else .inexact_residual_contour_filter;
    return if (q.linear_policy == .converged) .rational_filter else .inexact_rational_filter;
}
fn narrow(comptime T: type, real: []const Wide(T), imaginary: []const Wide(T), output: []T) !void {
    for (real, imaginary, output) |re, im, *out| out.* = try A.produced(T, .{ .re = @floatCast(re), .im = @floatCast(im) });
}
fn commit(comptime T: type, q: Request(T), staged: []const T) void {
    const n = q.action.rows;
    for (0..q.input.shape.column_count) |column| for (0..n) |row| {
        q.output.column(column).at(row).* = staged[column * n + row];
    };
}
fn commitZero(comptime T: type, q: Request(T)) void {
    if (q.output.shape.row_count == 0 or q.output.shape.column_count == 0) return;
    for (0..q.output.shape.column_count) |column| for (0..q.output.shape.row_count) |row| {
        q.output.column(column).at(row).* = S.zero(T);
    };
}
pub fn runAdmitted(comptime T: type, operation: *const E.context.Operation, q: Request(T), values: []T, wide: []Wide(T)) !Result(T) {
    _ = operation.borrowedContext();
    const sync = q.action.synchronization;
    try P.enter(T, sync, operation, algorithm(T, q));
    const length = try P.resolve(sync, operation, .{ .stage = .preflight }, preflight(T, q, values, wide));
    try P.check(sync, operation, .{ .stage = .global_dimension, .control = @intCast(q.global_dimension) });
    try P.check(sync, operation, .{ .stage = .panel_columns, .control = @intCast(q.input.shape.column_count) });
    try P.check(sync, operation, .{ .stage = .iteration_limit, .control = @intCast(q.max_iterations) });
    try P.check(sync, operation, .{ .stage = .relative_tolerance, .control = P.realBits(S.Real(T), q.relative_tolerance) });
    try P.check(sync, operation, .{ .stage = .absolute_tolerance, .control = P.realBits(S.Real(T), q.absolute_tolerance) });
    try P.check(sync, operation, .{ .stage = .reduction_backend, .control = @intFromBool(q.reduction != null) });
    try P.check(sync, operation, .{ .stage = .preconditioner_backend, .control = @intFromBool(q.preconditioner != null) });
    try P.coefficients(T, sync, operation, q.nodes);
    try P.coefficients(T, sync, operation, q.weights);
    if (q.ritz_correction) |correction| {
        try P.check(sync, operation, .{ .stage = .coefficients_count, .control = q.input.shape.column_count });
        for (correction.eigenvalues[0..q.input.shape.column_count], 0..) |lambda, index| try P.check(sync, operation, .{ .stage = .coefficient_real, .index = index, .control = P.realBits(S.Real(T), lambda) });
    }
    var result: Result(T) = .{};
    if (q.input.shape.column_count == 0 or q.global_dimension == 0 or q.nodes.len == 0) {
        try P.check(sync, operation, .{ .stage = .rational_commit });
        commitZero(T, q);
        return result;
    }
    const n = q.action.rows;
    const panel = n * q.input.shape.column_count;
    const input = values[0..panel];
    const output = values[panel .. 2 * panel];
    const end = (if (q.ritz_correction != null) @as(usize, 3) else 2) * panel;
    const basis = values[2 * panel .. end];
    const solution = values[end..][0..n];
    const linear_work = values[end + n .. length.elements];
    const real = wide[0..panel];
    const imaginary = wide[panel..length.wide_elements];
    @memset(real, 0);
    @memset(imaginary, 0);
    // Pack before any publication, so arbitrary regular input/output overlap is
    // supported. Logical input is ignored when every weight is exactly zero.
    var active = false;
    for (q.weights) |weight| active = active or !S.isZero(T, weight);
    if (active) for (0..q.input.shape.column_count) |column| for (0..n) |row| {
        input[column * n + row] = q.input.column(column).at(row);
        if (q.ritz_correction) |correction| basis[column * n + row] = correction.basis.column(column).at(row);
    };
    for (q.nodes, q.weights, 0..) |node, weight, node_index| {
        try P.check(sync, operation, .{ .stage = .rational_node, .index = @intCast(node_index) });
        if (S.isZero(T, weight)) continue;
        for (0..q.input.shape.column_count) |column| {
            try P.check(sync, operation, .{ .stage = .rational_column, .index = @intCast(column) });
            const linear = try B.Managed(E).runBorrowedAdmitted(T, operation, .{
                .action = q.action,
                .action_scale = .{ .re = -1, .im = 0 },
                .diagonal_shift = node,
                .preconditioner = q.preconditioner,
                .reduction = q.reduction,
                .rhs = input[column * n ..][0..n],
                .output = solution,
                .global_dimension = q.global_dimension,
                .max_iterations = q.max_iterations,
                .relative_tolerance = q.relative_tolerance,
                .absolute_tolerance = q.absolute_tolerance,
            }, linear_work);
            try P.resolve(sync, operation, .{ .stage = .local_arithmetic }, addCounts(T, &result, linear));
            try P.check(sync, operation, .{ .stage = .rational_system_status, .control = @backingInt(linear.status) });
            if (linear.status != .converged and !(q.linear_policy == .finite_iteration_limit and linear.status == .iteration_limit)) {
                result.status = .linear_stop;
                result.stopped_node = node_index;
                result.stopped_column = column;
                result.linear_status = linear.status;
                return result;
            }
            if (linear.status != .converged) result.inexact_systems = try P.resolve(sync, operation, .{ .stage = .local_arithmetic }, sum(result.inexact_systems, 1));
            result.completed_systems = try P.resolve(sync, operation, .{ .stage = .local_arithmetic }, sum(result.completed_systems, 1));
            if (q.ritz_correction) |correction| {
                try P.resolve(sync, operation, .{ .stage = .rational_accumulation }, accumulateCorrection(T, weight, node, correction.eigenvalues[column], basis[column * n ..][0..n], solution, real[column * n ..][0..n], imaginary[column * n ..][0..n]));
            } else try P.resolve(sync, operation, .{ .stage = .rational_accumulation }, accumulate(T, weight, solution, real[column * n ..][0..n], imaginary[column * n ..][0..n]));
        }
    }
    try P.resolve(sync, operation, .{ .stage = .local_arithmetic }, narrow(T, real, imaginary, output));
    try P.check(sync, operation, .{ .stage = .rational_commit });
    commit(T, q, output);
    return result;
}
pub fn Managed(comptime Execution: type) type {
    return struct {
        const C = Execution.context;
        pub fn runBorrowedAdmitted(comptime T: type, operation: *const C.Operation, q: Request(T), values: []T, wide: []Wide(T)) !Result(T) {
            return runAdmitted(T, operation, q, values, wide);
        }
        pub fn runScratchAdmitted(comptime T: type, operation: *const C.Operation, q: Request(T)) !Result(T) {
            _ = operation.borrowedContext();
            const sync = q.action.synchronization;
            try P.enter(T, sync, operation, algorithm(T, q));
            try P.resolve(sync, operation, .{ .stage = .preflight }, options(T, q));
            const length = try P.resolve(sync, operation, .{ .stage = .workspace }, workspaceLength(T, q));
            var values = try P.acquire(C, T, sync, operation, length.elements, .vector_allocation);
            defer values.deinit();
            var wide = try P.acquire(C, Wide(T), sync, operation, length.wide_elements, .real_allocation);
            defer wide.deinit();
            return runAdmitted(T, operation, q, values.lease.values, wide.lease.values);
        }
        pub fn runBorrowed(comptime T: type, owner: *C.Context, q: Request(T), values: []T, wide: []Wide(T)) !Result(T) {
            var operation = try P.begin(q.action.synchronization, owner);
            defer operation.end();
            return runAdmitted(T, &operation, q, values, wide);
        }
        pub fn runScratch(comptime T: type, owner: *C.Context, q: Request(T)) !Result(T) {
            var operation = try P.begin(q.action.synchronization, owner);
            defer operation.end();
            return runScratchAdmitted(T, &operation, q);
        }
    };
}
