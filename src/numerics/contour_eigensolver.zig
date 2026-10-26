// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Native Hermitian contour subspace iteration over borrowed operators/panels.
//! Only the projected subspace problem is dense. Full-space storage is O(N*m).
//! Default convergence tests fresh A*x residuals after global normalization:
//! ||A*x-lambda*x|| <= absolute + relative*max(1,abs(lambda)).
//! Optional trace convergence tests trace change divided by the interval scale
//! max(abs(lower),abs(upper)); this criterion alone does not certify residuals.
//! The closed interval selects returned Ritz pairs. Completeness still requires
//! a sufficiently rich starting subspace and adequate capacity. A full selected
//! capacity below N is reported separately; an absent Ritz pair never certifies
//! an empty spectrum. Errors preserve every public output. Controlled stops also
//! preserve outputs by default; an explicit policy can publish finite candidates.
//! All-Ritz output places selected pairs first, then the remaining sorted pairs.
//! Optional residual-inverse iterations apply the same contour to Ritz residuals,
//! retaining their basis contribution; no additional full-space owner is needed.
//! Shared operation, scratch, reductions and phase agreement retain ownership.
const std = @import("std");
const S = @import("../blas/core/shared/scalar.zig");
const D = @import("dense.zig");
const O = @import("operator.zig");
const Q = @import("contour_quadrature.zig");
const F = @import("rational_filter.zig");
const R = @import("rank_reduced_ritz.zig");
const Projection = @import("subspace_projection.zig");
const Red = @import("reductions.zig");
const Cx = @import("complex.zig");
const Store = @import("storage.zig");
const H = @import("../lapack/core/hermitian_eigen.zig");
const E = @import("zynum-execution");
const P = @import("synchronization.zig");

pub const Status = enum { converged, iteration_limit, capacity_limit, empty_subspace, linear_stop, ritz_stop };
pub const OutputSelection = enum { selected, all_ritz };
pub const LinearAccuracy = enum { fixed, residual_scaled };
pub const Convergence = enum { residual, trace };
pub const Method = enum { filtered_subspace, residual_inverse };
pub const Workspace = struct { elements: usize, real_elements: usize, wide_elements: usize, complex_elements: usize };
pub fn Request(comptime T: type) type {
    return struct {
        action: O.Operator(T),
        preconditioner: ?O.Operator(T) = null,
        reduction: ?Red.Reduction(T) = null,
        projection_reduction: ?Projection.Reduction(T) = null,
        initial: D.ConstPanel(T),
        vectors: D.Panel(T),
        eigenvalues: []S.Real(T),
        residuals: []S.Real(T),
        global_dimension: usize,
        contour: Q.Options(S.Real(T)),
        max_iterations: usize,
        relative_tolerance: S.Real(T),
        absolute_tolerance: S.Real(T) = 0,
        linear_max_iterations: usize,
        linear_relative_tolerance: S.Real(T),
        linear_absolute_tolerance: S.Real(T) = 0,
        linear_accuracy: LinearAccuracy = .fixed,
        allow_inexact_iteration_limit: bool = false,
        rank_relative_tolerance: ?S.Real(T) = null,
        rank_absolute_tolerance: S.Real(T) = 0,
        eigen_options: H.Options = .{},
        output_selection: OutputSelection = .selected,
        publish_finite_stops: bool = false,
        convergence: Convergence = .residual,
        trace_tolerance: S.Real(T) = 0,
        method: Method = .filtered_subspace,
    };
}
pub fn Result(comptime T: type) type {
    return struct {
        status: Status,
        iterations: usize = 0,
        rank: usize = 0,
        found: usize = 0,
        converged_pairs: usize = 0,
        published: usize = 0,
        completed_systems: usize = 0,
        linear_iterations: usize = 0,
        inexact_systems: usize = 0,
        action_count: usize = 0,
        max_residual_norm: S.Real(T) = 0,
        trace_change: S.Real(T) = 1,
        linear: ?F.Result(T) = null,
        ritz: ?R.Result = null,
    };
}
fn add(a: usize, b: usize) !usize {
    return std.math.add(usize, a, b) catch error.SizeOverflow;
}
fn mul(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch error.SizeOverflow;
}
fn panel(comptime T: type, values: []T, rows: usize, columns: usize) D.Panel(T) {
    return .{ .values = values, .shape = .{ .row_count = rows, .column_count = columns, .leading_dimension = rows, .layout = .column_major } };
}
fn small(comptime T: type, values: []T, stride: usize, order: usize) D.Matrix(T) {
    return .{ .values = values, .row_count = order, .column_count = order, .leading_dimension = stride };
}
fn filter(comptime T: type, q: Request(T), input: []T, output: []T, columns: usize, nodes: []const T, weights: []const T) F.Request(T) {
    return .{ .action = q.action, .preconditioner = q.preconditioner, .reduction = q.reduction, .input = panel(T, input, q.action.rows, columns).asConst(), .output = panel(T, output, q.action.rows, columns), .nodes = nodes, .weights = weights, .global_dimension = q.global_dimension, .max_iterations = q.linear_max_iterations, .relative_tolerance = q.linear_relative_tolerance, .absolute_tolerance = q.linear_absolute_tolerance, .linear_policy = if (q.allow_inexact_iteration_limit) .finite_iteration_limit else .converged };
}
pub fn workspaceLength(comptime T: type, q: Request(T)) !Workspace {
    comptime Cx.expectComplex(T);
    const m = q.initial.shape.column_count;
    const count = try Q.nodeCount(T, q.contour);
    if (q.global_dimension == 0 or m == 0 or q.max_iterations == 0) return .{ .elements = 0, .real_elements = 0, .wide_elements = 0, .complex_elements = 0 };
    const block = try mul(q.action.rows, m);
    const square = try mul(m, m);
    // Rational-filter storage depends on node presence, not node count.
    const stub = [_]T{S.zero(T)};
    var filter_request = filter(T, q, &.{}, &.{}, m, &stub, &stub);
    if (q.method == .residual_inverse) filter_request.ritz_correction = .{ .basis = panel(T, &.{}, q.action.rows, m).asConst(), .eigenvalues = &.{} };
    const f = try F.workspaceLength(T, filter_request);
    const r = try R.workspaceLength(T, m);
    const leaf = @max(f.elements, try q.action.panelWorkspaceElements(S.one(T), m), try Projection.workspaceLength(T, q.action.rows, m), r.elements);
    const elements = try add(try add(try mul(block, 4), try mul(count, 2)), try add(try mul(square, 3), leaf));
    const reals = try mul(m, 4);
    _ = try Store.byteLength(elements, @sizeOf(T));
    _ = try Store.byteLength(reals, @sizeOf(S.Real(T)));
    _ = try Store.byteLength(r.wide_elements, @sizeOf(H.WorkspaceScalar(T)));
    return .{ .elements = elements, .real_elements = reals, .wide_elements = f.wide_elements, .complex_elements = r.wide_elements };
}
fn range(values: anytype) !Store.ByteRange {
    return Store.byteRange(@intFromPtr(values.ptr), try Store.byteLength(values.len, @sizeOf(@typeInfo(@TypeOf(values)).pointer.child)));
}
fn separate(a: anytype, b: anytype) !void {
    if (Store.overlaps(try range(a), try range(b))) return error.AliasingNotAllowed;
}
fn nonnegative(value: anytype) bool {
    return std.math.isFinite(value) and value >= 0;
}
fn options(comptime T: type, q: Request(T)) !void {
    comptime Cx.expectComplex(T);
    const n = q.action.rows;
    const m = q.initial.shape.column_count;
    _ = try q.initial.check();
    const output = q.vectors.values[0..try q.vectors.check()];
    if (q.action.columns != n or n > q.global_dimension or m > q.global_dimension or
        q.initial.shape.row_count != n or q.vectors.shape.row_count != n or q.vectors.shape.column_count != m or
        q.eigenvalues.len < m or q.residuals.len < m or (m == 0 and q.global_dimension != 0)) return error.DimensionMismatch;
    if (n != q.global_dimension and (q.reduction == null or q.projection_reduction == null)) return error.MissingGlobalReduction;
    if (!nonnegative(q.relative_tolerance) or !nonnegative(q.absolute_tolerance) or
        !nonnegative(q.linear_relative_tolerance) or !nonnegative(q.linear_absolute_tolerance) or !nonnegative(q.rank_absolute_tolerance) or !nonnegative(q.trace_tolerance)) return error.InvalidOptions;
    if (q.rank_relative_tolerance) |t| if (!nonnegative(t) or t >= 1) return error.InvalidOptions;
    if (q.eigen_options.relative_tolerance) |t| if (!nonnegative(t)) return error.InvalidOptions;
    if (q.preconditioner) |inverse| {
        if (inverse.rows != n or inverse.columns != n) return error.DimensionMismatch;
        if (inverse.synchronization) |sync| {
            const parent = q.action.synchronization orelse return error.InvalidOptions;
            if (sync.context != parent.context or sync.agree_fn != parent.agree_fn) return error.InvalidOptions;
        }
    }
    _ = try Q.point(T, q.contour, 0);
    try separate(output, q.eigenvalues[0..m]);
    try separate(output, q.residuals[0..m]);
    try separate(q.eigenvalues[0..m], q.residuals[0..m]);
    if (q.max_iterations != 0) for (0..m) |column| for (0..n) |row| {
        if (!Red.finite(T, q.initial.column(column).at(row))) return error.NonFiniteInput;
    };
}
/// Frontends can validate the canonical request before reserving publication
/// buffers. This performs no admission, allocation, action or communication.
pub fn validateRequest(comptime T: type, q: Request(T)) !void {
    try options(T, q);
}
fn preflight(comptime T: type, q: Request(T), work: []T, real: []S.Real(T), wide: []Cx.Wide(T), extra: []H.WorkspaceScalar(T)) !Workspace {
    try options(T, q);
    const w = try workspaceLength(T, q);
    if (work.len < w.elements or real.len < w.real_elements or wide.len < w.wide_elements or extra.len < w.complex_elements) return error.WorkspaceTooSmall;
    const a = work[0..w.elements];
    const b = real[0..w.real_elements];
    const c = wide[0..w.wide_elements];
    const d = extra[0..w.complex_elements];
    try separate(a, b);
    try separate(a, c);
    try separate(a, d);
    try separate(b, c);
    try separate(b, d);
    try separate(c, d);
    for ([_][]const T{ q.initial.values[0..try q.initial.check()], q.vectors.values[0..try q.vectors.check()] }) |v| {
        try separate(v, a);
        try separate(v, b);
        try separate(v, c);
        try separate(v, d);
    }
    for ([_][]const S.Real(T){ q.eigenvalues[0..q.initial.shape.column_count], q.residuals[0..q.initial.shape.column_count] }) |v| {
        try separate(v, a);
        try separate(v, b);
        try separate(v, c);
        try separate(v, d);
    }
    return w;
}
fn controls(comptime T: type, op: *const E.context.Operation, q: Request(T)) !void {
    const sync = q.action.synchronization;
    try P.check(sync, op, .{ .stage = .global_dimension, .control = @intCast(q.global_dimension) });
    try P.check(sync, op, .{ .stage = .panel_columns, .control = @intCast(q.initial.shape.column_count) });
    const raw = [_]u64{ @intCast(q.max_iterations), @intCast(q.linear_max_iterations), @intCast(q.contour.half_nodes), @backingInt(q.contour.rule), @intCast(q.contour.gauss_max_iterations), @intFromBool(q.reduction != null), @intFromBool(q.projection_reduction != null), @intFromBool(q.preconditioner != null), @intFromBool(q.rank_relative_tolerance != null), @intFromBool(q.eigen_options.relative_tolerance != null), @intCast(q.eigen_options.max_sweeps), @backingInt(q.output_selection), @intFromBool(q.publish_finite_stops), @backingInt(q.linear_accuracy), @intFromBool(q.allow_inexact_iteration_limit), @backingInt(q.convergence), @backingInt(q.method) };
    for (raw, 0..) |value, i| try P.check(sync, op, .{ .stage = .contour_control, .index = i, .control = value });
    const values = [_]S.Real(T){ q.contour.lower, q.contour.upper, q.contour.vertical_ratio, q.relative_tolerance, q.absolute_tolerance, q.linear_relative_tolerance, q.linear_absolute_tolerance, q.rank_relative_tolerance orelse 0, q.rank_absolute_tolerance, q.trace_tolerance };
    for (values, raw.len..) |value, i| try P.check(sync, op, .{ .stage = .contour_control, .index = i, .control = P.realBits(S.Real(T), value) });
    try P.check(sync, op, .{ .stage = .contour_control, .index = raw.len + values.len, .control = P.realBits(f64, q.eigen_options.relative_tolerance orelse 0) });
}
fn divide(comptime T: type, values: []T, scale: S.Real(T)) !void {
    for (values) |*value| value.* = Cx.narrow(T, .{ .re = @as(Cx.Wide(T), value.re) / scale, .im = @as(Cx.Wide(T), value.im) / scale }) orelse return error.NonFiniteResult;
}
fn normalize(comptime T: type, op: *const E.context.Operation, q: Request(T), values: []T, columns: usize, permit_zero: bool) !void {
    const n = q.action.rows;
    for (0..columns) |column| {
        const x = values[column * n ..][0..n];
        const norm = try P.resolve(q.action.synchronization, op, .{ .stage = .contour_normalization, .index = column }, Red.norm(T, op, q.reduction, x));
        try P.check(q.action.synchronization, op, .{ .stage = .contour_normalization, .index = column, .control = P.realBits(S.Real(T), norm) });
        if (norm == 0) {
            if (!permit_zero) return error.ZeroRitzVector;
        } else try P.resolve(q.action.synchronization, op, .{ .stage = .local_arithmetic, .index = column }, divide(T, x, norm));
    }
}
fn rotate(comptime T: type, basis: []const T, coefficients: []const T, output: []T, n: usize, m: usize, rank: usize, stride: usize) !void {
    for (0..rank) |column| for (0..n) |row| {
        var total: Cx.WideComplex(T) = .{ .re = 0, .im = 0 };
        for (0..m) |k| {
            const a = Cx.widen(T, basis[row + k * n]);
            const b = Cx.widen(T, coefficients[k + column * stride]);
            total.re += a.re * b.re - a.im * b.im;
            total.im += a.re * b.im + a.im * b.re;
        }
        output[row + column * n] = Cx.narrow(T, total) orelse return error.NonFiniteResult;
    };
}
fn residual(comptime T: type, x: []const T, ax: []const T, lambda: S.Real(T), out: []T) !void {
    for (x, ax, out) |a, b, *r| r.* = Cx.narrow(T, .{ .re = @as(Cx.Wide(T), b.re) - @as(Cx.Wide(T), lambda) * a.re, .im = @as(Cx.Wide(T), b.im) - @as(Cx.Wide(T), lambda) * a.im }) orelse return error.NonFiniteResult;
}
fn counts(comptime T: type, result: *Result(T), linear: F.Result(T)) !void {
    result.completed_systems = try add(result.completed_systems, linear.completed_systems);
    result.linear_iterations = try add(result.linear_iterations, linear.iterations);
    result.inexact_systems = try add(result.inexact_systems, linear.inexact_systems);
    result.action_count = try add(result.action_count, linear.action_count);
}
fn selected(comptime T: type, q: Request(T), value: S.Real(T)) bool {
    return value >= q.contour.lower and value <= q.contour.upper;
}
fn applyPanel(comptime T: type, op: *const E.context.Operation, q: Request(T), input: []T, output: []T, columns: usize, scratch: []T) !void {
    // A globally nonempty problem may have an empty local shard. The generic
    // panel convenience path permits skipping zero-row unsynchronized actions;
    // this algorithm explicitly retains every vector callback in that case.
    if (q.action.rows == 0 and q.action.synchronization == null and q.action.apply_panel_fn == null) {
        for (0..columns) |_| try q.action.applyAdmitted(op, input[0..0], output[0..0], scratch);
    } else try q.action.applyPanelAdmitted(op, S.one(T), panel(T, input, q.action.rows, columns).asConst(), S.zero(T), panel(T, output, q.action.rows, columns), scratch);
}
fn commit(comptime T: type, q: Request(T), values: []const S.Real(T), errors: []const S.Real(T), vectors: []const T, rank: usize) usize {
    var target: usize = 0;
    for (0..2) |pass| {
        if (pass == 1 and q.output_selection == .selected) break;
        for (0..rank) |column| if (selected(T, q, values[column]) == (pass == 0)) {
            q.eigenvalues[target] = values[column];
            q.residuals[target] = errors[column];
            for (0..q.action.rows) |row| q.vectors.column(target).at(row).* = vectors[column * q.action.rows + row];
            target += 1;
        };
    }
    return target;
}

pub fn runAdmitted(comptime T: type, op: *const E.context.Operation, q: Request(T), work: []T, real: []S.Real(T), wide: []Cx.Wide(T), extra: []H.WorkspaceScalar(T)) !Result(T) {
    _ = op.borrowedContext();
    const sync = q.action.synchronization;
    try P.enter(T, sync, op, .contour_eigensolver);
    const w = try P.resolve(sync, op, .{ .stage = .preflight }, preflight(T, q, work, real, wide, extra));
    try controls(T, op, q);
    var result: Result(T) = .{ .status = .iteration_limit };
    if (q.global_dimension == 0) return .{ .status = .converged };
    if (q.max_iterations == 0) return result;
    const n = q.action.rows;
    const capacity = q.initial.shape.column_count;
    const block = n * capacity;
    const square = capacity * capacity;
    const count = try Q.nodeCount(T, q.contour);
    const basis = work[0..block];
    const filtered = work[block .. 2 * block];
    const applied = work[2 * block .. 3 * block];
    const vectors = work[3 * block .. 4 * block];
    const nodes = work[4 * block ..][0..count];
    const weights = work[4 * block + count ..][0..count];
    const matrices = work[4 * block + 2 * count ..];
    const h = matrices[0..square];
    const s = matrices[square .. 2 * square];
    const coefficients = matrices[2 * square .. 3 * square];
    const scratch = matrices[3 * square .. w.elements - 4 * block - 2 * count];
    const values = real[0..capacity];
    const errors = real[capacity .. 2 * capacity];
    const roots = real[2 * capacity .. w.real_elements];
    try P.resolve(sync, op, .{ .stage = .local_arithmetic }, Q.populate(T, q.contour, nodes, weights));
    for (0..capacity) |column| for (0..n) |row| {
        basis[row + column * n] = q.initial.column(column).at(row);
    };
    var active = capacity;
    var linear_tolerance = q.linear_relative_tolerance;
    var previous_trace: ?Cx.Wide(T) = null;
    for (0..q.max_iterations) |iteration| {
        result.iterations = iteration + 1;
        try P.check(sync, op, .{ .stage = .contour_iteration, .index = iteration, .control = active });
        var linear_request = filter(T, q, if (q.method == .residual_inverse and iteration != 0) applied else basis, filtered, active, nodes, weights);
        if (q.method == .residual_inverse and iteration != 0) linear_request.ritz_correction = .{ .basis = panel(T, basis, n, active).asConst(), .eigenvalues = values[0..active] };
        linear_request.relative_tolerance = linear_tolerance;
        const linear = try F.runAdmitted(T, op, linear_request, scratch, wide[0..w.wide_elements]);
        try P.resolve(sync, op, .{ .stage = .local_arithmetic }, counts(T, &result, linear));
        result.linear = linear;
        if (linear.status != .success) {
            result.status = .linear_stop;
            return result;
        }
        try normalize(T, op, q, filtered, active, true);
        try applyPanel(T, op, q, filtered, applied, active, scratch);
        result.action_count = try P.resolve(sync, op, .{ .stage = .local_arithmetic }, add(result.action_count, if (q.action.apply_panel_fn != null) 1 else active));
        try Projection.runAdmitted(T, op, .{ .basis = panel(T, filtered, n, active).asConst(), .applied = panel(T, applied, n, active).asConst(), .hamiltonian = .{ .values = h, .shape = .{ .row_count = active, .column_count = active, .leading_dimension = capacity, .layout = .column_major } }, .overlap = .{ .values = s, .shape = .{ .row_count = active, .column_count = active, .leading_dimension = capacity, .layout = .column_major } }, .reduction = q.projection_reduction, .synchronization = sync }, scratch);
        const ritz = try P.resolve(sync, op, .{ .stage = .local_arithmetic }, R.runAdmitted(T, op, .{ .hamiltonian = small(T, h, capacity, active).asConst(), .overlap = small(T, s, capacity, active).asConst(), .coefficients = small(T, coefficients, capacity, active), .eigenvalues = values, .rank_relative_tolerance = q.rank_relative_tolerance, .rank_absolute_tolerance = q.rank_absolute_tolerance, .eigen_options = q.eigen_options }, scratch, roots, extra[0..w.complex_elements]));
        result.ritz = ritz;
        result.rank = ritz.rank;
        try P.check(sync, op, .{ .stage = .contour_status, .control = @backingInt(ritz.status) });
        try P.check(sync, op, .{ .stage = .contour_rank, .control = ritz.rank });
        if (ritz.status != .converged) {
            result.status = .ritz_stop;
            return result;
        }
        if (ritz.rank == 0) {
            result.status = .empty_subspace;
            return result;
        }
        for (values[0..ritz.rank], 0..) |value, i| try P.check(sync, op, .{ .stage = .contour_eigenvalue, .index = i, .control = P.realBits(S.Real(T), value) });
        try P.resolve(sync, op, .{ .stage = .local_arithmetic }, rotate(T, filtered, coefficients, vectors, n, active, ritz.rank, capacity));
        try normalize(T, op, q, vectors, ritz.rank, false);
        // Fresh callbacks, rather than rotating A*Q, verify actual final vectors.
        try applyPanel(T, op, q, vectors, applied, ritz.rank, scratch);
        result.action_count = try P.resolve(sync, op, .{ .stage = .local_arithmetic }, add(result.action_count, if (q.action.apply_panel_fn != null) 1 else ritz.rank));
        result.found = 0;
        result.converged_pairs = 0;
        result.max_residual_norm = 0;
        var worst_relative_residual: Cx.Wide(T) = 0;
        var worst_selected_residual: Cx.Wide(T) = 0;
        var trace: Cx.Wide(T) = 0;
        for (0..ritz.rank) |column| {
            const x = vectors[column * n ..][0..n];
            const ax = applied[column * n ..][0..n];
            const residual_vector = if (q.method == .residual_inverse) ax else filtered[0..n];
            try P.resolve(sync, op, .{ .stage = .local_arithmetic, .index = column }, residual(T, x, ax, values[column], residual_vector));
            errors[column] = try P.resolve(sync, op, .{ .stage = .contour_residual, .index = column }, Red.norm(T, op, q.reduction, residual_vector));
            try P.check(sync, op, .{ .stage = .contour_residual, .index = column, .control = P.realBits(S.Real(T), errors[column]) });
            worst_relative_residual = @max(worst_relative_residual, @as(Cx.Wide(T), errors[column]) / @max(@as(Cx.Wide(T), 1), @abs(@as(Cx.Wide(T), values[column]))));
            if (selected(T, q, values[column])) {
                trace += values[column];
                worst_selected_residual = @max(worst_selected_residual, @as(Cx.Wide(T), errors[column]) / @max(@as(Cx.Wide(T), 1), @abs(@as(Cx.Wide(T), values[column]))));
                result.found += 1;
                result.max_residual_norm = @max(result.max_residual_norm, errors[column]);
                const threshold = @as(Cx.Wide(T), q.absolute_tolerance) + @as(Cx.Wide(T), q.relative_tolerance) * @max(@as(Cx.Wide(T), 1), @abs(@as(Cx.Wide(T), values[column])));
                if (@as(Cx.Wide(T), errors[column]) <= threshold) result.converged_pairs += 1;
            }
        }
        if (previous_trace) |old_trace| result.trace_change = @floatCast(@abs(trace - old_trace) / @max(@abs(@as(Cx.Wide(T), q.contour.lower)), @abs(@as(Cx.Wide(T), q.contour.upper))));
        previous_trace = trace;
        const converged = if (q.convergence == .residual) result.found == result.converged_pairs else result.trace_change < q.trace_tolerance;
        if (result.found != 0 and converged) {
            if (result.found == capacity and capacity < q.global_dimension) {
                result.status = .capacity_limit;
                if (q.publish_finite_stops) {
                    try P.check(sync, op, .{ .stage = .commit });
                    result.published = commit(T, q, values, errors, vectors, ritz.rank);
                }
                return result;
            }
            try P.check(sync, op, .{ .stage = .commit });
            result.published = commit(T, q, values, errors, vectors, ritz.rank);
            result.status = .converged;
            return result;
        }
        if (q.linear_accuracy == .residual_scaled) {
            const upper: Cx.Wide(T) = q.linear_relative_tolerance;
            const floor: Cx.Wide(T) = std.math.floatEps(S.Real(T)) * 8;
            linear_tolerance = @floatCast(@min(upper, @max(floor, upper * (if (result.found != 0) worst_selected_residual else worst_relative_residual))));
        }
        @memcpy(basis[0 .. n * ritz.rank], vectors[0 .. n * ritz.rank]);
        active = ritz.rank;
    }
    if (q.publish_finite_stops) {
        try P.check(sync, op, .{ .stage = .commit });
        result.published = commit(T, q, values, errors, vectors, result.rank);
    }
    return result;
}
pub fn Managed(comptime Execution: type) type {
    return struct {
        const C = Execution.context;
        pub fn runBorrowedAdmitted(comptime T: type, op: *const C.Operation, q: Request(T), values: []T, real: []S.Real(T), wide: []Cx.Wide(T), extra: []H.WorkspaceScalar(T)) !Result(T) {
            return runAdmitted(T, op, q, values, real, wide, extra);
        }
        pub fn runScratchAdmitted(comptime T: type, op: *const C.Operation, q: Request(T)) !Result(T) {
            try P.enter(T, q.action.synchronization, op, .contour_eigensolver);
            try P.resolve(q.action.synchronization, op, .{ .stage = .preflight }, options(T, q));
            const w = try P.resolve(q.action.synchronization, op, .{ .stage = .workspace }, workspaceLength(T, q));
            var values = try P.acquire(C, T, q.action.synchronization, op, w.elements, .vector_allocation);
            defer values.deinit();
            var real = try P.acquire(C, S.Real(T), q.action.synchronization, op, w.real_elements, .real_allocation);
            defer real.deinit();
            var wide = try P.acquire(C, Cx.Wide(T), q.action.synchronization, op, w.wide_elements, .real_allocation);
            defer wide.deinit();
            var extra = try P.acquire(C, H.WorkspaceScalar(T), q.action.synchronization, op, w.complex_elements, .vector_allocation);
            defer extra.deinit();
            return runAdmitted(T, op, q, values.lease.values, real.lease.values, wide.lease.values, extra.lease.values);
        }
        pub fn runBorrowed(comptime T: type, owner: *C.Context, q: Request(T), values: []T, real: []S.Real(T), wide: []Cx.Wide(T), extra: []H.WorkspaceScalar(T)) !Result(T) {
            var op = try P.begin(q.action.synchronization, owner);
            defer op.end();
            return runAdmitted(T, &op, q, values, real, wide, extra);
        }
        pub fn runScratch(comptime T: type, owner: *C.Context, q: Request(T)) !Result(T) {
            var op = try P.begin(q.action.synchronization, owner);
            defer op.end();
            return runScratchAdmitted(T, &op, q);
        }
    };
}
