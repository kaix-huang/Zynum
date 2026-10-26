// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Synchronous checked Chebyshev actions on the shared admitted Operator.
//! Caller supplies the affine action and coefficients; no rescaling policy,
//! communicator, allocator, thread manager or persistent handle is introduced.
//! Opaque numerical action data must remain fixed and callback code valid through
//! return; output must not alias live action state. Concurrent use requires a callback
//! that supports it, as with the shared Operator contract.
const std = @import("std");
const scalar = @import("../blas/core/shared/scalar.zig");
const storage = @import("storage.zig");
const operators = @import("operator.zig");
const dense = @import("dense.zig");
const execution = @import("zynum-execution");
const synchronization = @import("synchronization.zig");

pub fn Request(comptime T: type) type {
    return struct {
        action: operators.Operator(T),
        input: []const T,
        coefficients: []const T,
        output: []T,
    };
}
fn finite(comptime T: type, value: T) bool {
    return if (comptime scalar.isComplex(T)) std.math.isFinite(value.re) and std.math.isFinite(value.im) else std.math.isFinite(value);
}
fn checkFinite(comptime T: type, values: []const T) !void {
    for (values) |value| if (!finite(T, value)) return error.NonFiniteInput;
}
fn checkProduced(comptime T: type, values: []const T) !void {
    for (values) |value| if (!finite(T, value)) return error.NonFiniteResult;
}
fn range(comptime T: type, values: anytype) !storage.ByteRange {
    return storage.byteRange(@intFromPtr(values.ptr), try storage.byteLength(values.len, @sizeOf(T)));
}
fn noOverlap(comptime T: type, first: anytype, second: anytype) !void {
    if (storage.overlaps(try range(T, first), try range(T, second))) return error.AliasingNotAllowed;
}
fn square(comptime T: type, action: operators.Operator(T), input: []const T, output: []T) !void {
    if (action.rows != action.columns or input.len != action.columns or output.len != action.rows) return error.DimensionMismatch;
}
fn required(comptime T: type, action: operators.Operator(T), vectors: usize) !usize {
    const count = std.math.mul(usize, action.rows, vectors) catch return error.SizeOverflow;
    const used = std.math.add(usize, count, action.workspace_elements) catch return error.SizeOverflow;
    _ = try storage.byteLength(used, @sizeOf(T));
    return used;
}
pub fn stepWorkspaceLength(comptime T: type, action: operators.Operator(T)) !usize {
    return required(T, action, 1);
}
pub fn seriesWorkspaceLength(comptime T: type, action: operators.Operator(T)) !usize {
    return required(T, action, 4);
}
// Widen products and the recurrence subtraction before narrowing. A finite
// final value is not rejected just because a native intermediate overflows.
fn linear(comptime T: type, a: T, x: T, b: T, y: T) !T {
    return @import("recurrence_arithmetic.zig").linear(T, a, x, b, y);
}

/// next=2*A*current-previous. Required scratch prefix holds callback output;
/// callback errors and nonfinite results leave next untouched. Scratch may change.
/// Input views may overlap each other; output and scratch cannot overlap inputs.
pub fn stepAdmitted(comptime T: type, operation: *const execution.context.Operation, action: operators.Operator(T), previous: []const T, current: []const T, next: []T, workspace: []T) !void {
    _ = operation.borrowedContext();
    try synchronization.enter(T, action.synchronization, operation, .chebyshev_step);
    const work = try synchronization.resolve(action.synchronization, operation, .{ .stage = .preflight }, stepPreflight(T, action, previous, current, next, workspace));
    const product = work[action.workspace_elements..];
    try action.applyAdmitted(operation, current, product, work[0..action.workspace_elements]);
    try synchronization.resolve(action.synchronization, operation, .{ .stage = .local_arithmetic }, stepArithmetic(T, product, previous));
    try synchronization.check(action.synchronization, operation, .{ .stage = .commit });
    @memcpy(next, product);
}
fn stepArithmetic(comptime T: type, product: []T, previous: []const T) !void {
    try checkProduced(T, product);
    for (product, previous) |*value, old| value.* = try linear(T, scalar.realScalar(T, 2), value.*, scalar.neg(T, scalar.one(T)), old);
}
fn stepPreflight(comptime T: type, action: operators.Operator(T), previous: []const T, current: []const T, next: []T, workspace: []T) ![]T {
    try square(T, action, current, next);
    if (previous.len != action.rows) return error.DimensionMismatch;
    const used = try stepWorkspaceLength(T, action);
    if (workspace.len < used) return error.WorkspaceTooSmall;
    const work = workspace[0..used];
    try noOverlap(T, previous, next);
    try noOverlap(T, current, next);
    try noOverlap(T, previous, work);
    try noOverlap(T, current, work);
    try noOverlap(T, next, work);
    try checkFinite(T, previous);
    try checkFinite(T, current);
    return work;
}

/// Sum c[k]*T_k(A)*input for a nonempty coefficient array. No endpoint halving
/// is implicit. One operation covers every action; input/coefficients remain
/// borrowed and immutable through return. Final output commits only on success.
/// Even zero-local-size actions participate once per requested order so a
/// distributed caller's callbacks keep the same collective sequence on all ranks.
pub fn seriesAdmitted(comptime T: type, operation: *const execution.context.Operation, request: Request(T), workspace: []T) !void {
    _ = operation.borrowedContext();
    const action = request.action;
    try synchronization.enter(T, action.synchronization, operation, .chebyshev_series);
    const work = try synchronization.resolve(action.synchronization, operation, .{ .stage = .preflight }, seriesPreflight(T, request, workspace));
    try synchronization.coefficients(T, action.synchronization, operation, request.coefficients);
    const n = action.rows;
    const callback_work = work[0..action.workspace_elements];
    const vectors = work[action.workspace_elements..];
    var previous = vectors[0..n];
    var current = vectors[n..][0..n];
    var next = vectors[2 * n ..][0..n];
    const sum = vectors[3 * n ..][0..n];
    @memcpy(previous, request.input);
    try synchronization.resolve(action.synchronization, operation, .{ .stage = .local_arithmetic }, initialSum(T, sum, previous, request.coefficients[0]));
    for (request.coefficients[1..], 1..) |coefficient, order| {
        try action.applyAdmitted(operation, if (order == 1) previous else current, next, callback_work);
        try synchronization.resolve(action.synchronization, operation, .{ .stage = .local_arithmetic, .index = @intCast(order) }, seriesArithmetic(T, next, previous, sum, coefficient, order));
        if (order > 1) std.mem.swap([]T, &previous, &current);
        std.mem.swap([]T, &current, &next);
    }
    try synchronization.check(action.synchronization, operation, .{ .stage = .commit });
    @memcpy(request.output, sum);
}
fn initialSum(comptime T: type, sum: []T, previous: []const T, coefficient: T) !void {
    for (sum, previous) |*value, x| value.* = try linear(T, coefficient, x, scalar.zero(T), scalar.zero(T));
}
fn seriesArithmetic(comptime T: type, next: []T, previous: []const T, sum: []T, coefficient: T, order: usize) !void {
    try checkProduced(T, next);
    if (order > 1) for (next, previous) |*value, old| {
        value.* = try linear(T, scalar.realScalar(T, 2), value.*, scalar.neg(T, scalar.one(T)), old);
    };
    for (sum, next) |*value, state| value.* = try linear(T, scalar.one(T), value.*, coefficient, state);
}
fn seriesPreflight(comptime T: type, request: Request(T), workspace: []T) ![]T {
    const action = request.action;
    try square(T, action, request.input, request.output);
    if (request.coefficients.len == 0) return error.InvalidOptions;
    const used = try seriesWorkspaceLength(T, action);
    if (workspace.len < used) return error.WorkspaceTooSmall;
    const work = workspace[0..used];
    try noOverlap(T, request.input, request.output);
    try noOverlap(T, request.coefficients, request.output);
    try noOverlap(T, request.input, work);
    try noOverlap(T, request.coefficients, work);
    try noOverlap(T, request.output, work);
    try checkFinite(T, request.input);
    try checkFinite(T, request.coefficients);
    return work;
}

pub fn PanelRequest(comptime T: type) type {
    return struct { action: operators.Operator(T), input: dense.ConstPanel(T), coefficients: []const T, output: dense.Panel(T) };
}
pub fn panelWorkspaceLength(comptime T: type, action: operators.Operator(T), columns: usize) !usize {
    const count = std.math.mul(usize, action.rows, columns) catch return error.SizeOverflow;
    const vectors = std.math.mul(usize, count, 4) catch return error.SizeOverflow;
    const callback = try action.panelWorkspaceElements(scalar.one(T), columns);
    const used = std.math.add(usize, vectors, callback) catch return error.SizeOverflow;
    _ = try storage.byteLength(used, @sizeOf(T));
    return used;
}
fn packedPanel(comptime T: type, values: []T, rows: usize, columns: usize) dense.Panel(T) {
    return .{ .values = values, .shape = .{ .row_count = rows, .column_count = columns, .leading_dimension = rows, .layout = .column_major } };
}
/// Same coefficients on each column, with existing fused SpMM/panel callbacks.
/// Only logical input/output elements participate; padding and storage tails
/// are preserved. Empty-action callback behavior is inherited from Operator.
pub fn panelAdmitted(comptime T: type, operation: *const execution.context.Operation, request: PanelRequest(T), workspace: []T) !void {
    _ = operation.borrowedContext();
    const action = request.action;
    const rows = action.rows;
    const columns = request.input.shape.column_count;
    try synchronization.enter(T, action.synchronization, operation, .chebyshev_panel);
    const work = try synchronization.resolve(action.synchronization, operation, .{ .stage = .preflight }, panelPreflight(T, request, workspace));
    try synchronization.check(action.synchronization, operation, .{ .stage = .panel_columns, .control = @intCast(columns) });
    try synchronization.coefficients(T, action.synchronization, operation, request.coefficients);
    const callback_size = try action.panelWorkspaceElements(scalar.one(T), columns);
    const callback_work = work[0..callback_size];
    const vectors = work[callback_size..];
    const count = rows * columns;
    var previous = vectors[0..count];
    var current = vectors[count..][0..count];
    var next = vectors[2 * count ..][0..count];
    const sum = vectors[3 * count ..][0..count];
    for (0..columns) |j| {
        const column = request.input.column(j);
        for (0..rows) |i| previous[i + j * rows] = column.at(i);
    }
    try synchronization.resolve(action.synchronization, operation, .{ .stage = .local_arithmetic }, initialSum(T, sum, previous, request.coefficients[0]));
    for (request.coefficients[1..], 1..) |coefficient, order| {
        try action.applyPanelAdmitted(operation, scalar.one(T), packedPanel(T, if (order == 1) previous else current, rows, columns).asConst(), scalar.zero(T), packedPanel(T, next, rows, columns), callback_work);
        try synchronization.resolve(action.synchronization, operation, .{ .stage = .local_arithmetic, .index = @intCast(order) }, seriesArithmetic(T, next, previous, sum, coefficient, order));
        if (order > 1) std.mem.swap([]T, &previous, &current);
        std.mem.swap([]T, &current, &next);
    }
    try synchronization.check(action.synchronization, operation, .{ .stage = .commit });
    for (0..columns) |j| {
        const column = request.output.column(j);
        for (0..rows) |i| column.at(i).* = sum[i + j * rows];
    }
}
fn panelPreflight(comptime T: type, request: PanelRequest(T), workspace: []T) ![]T {
    const action = request.action;
    const rows = action.rows;
    const columns = request.input.shape.column_count;
    const x_used = try request.input.check();
    const y_used = try request.output.check();
    if (action.rows != action.columns or request.input.shape.row_count != action.columns or request.output.shape.row_count != rows or request.output.shape.column_count != columns) return error.DimensionMismatch;
    if (request.coefficients.len == 0) return error.InvalidOptions;
    const used = try panelWorkspaceLength(T, action, columns);
    if (workspace.len < used) return error.WorkspaceTooSmall;
    const work = workspace[0..used];
    const input = request.input.values[0..x_used];
    const output = request.output.values[0..y_used];
    try noOverlap(T, input, output);
    try noOverlap(T, request.coefficients, output);
    try noOverlap(T, input, work);
    try noOverlap(T, request.coefficients, work);
    try noOverlap(T, output, work);
    try checkFinite(T, request.coefficients);
    for (0..columns) |j| {
        const column = request.input.column(j);
        for (0..rows) |i| {
            const x = column.at(i);
            if (!finite(T, x)) return error.NonFiniteInput;
        }
    }
    return work;
}

/// Different coefficient columns share a single recurrence on one input.
/// Counts are native-width lengths; unused coefficient rows and physical panel
/// padding are not read. There is no implicit endpoint halving.
pub fn SeriesSetRequest(comptime T: type) type {
    return struct {
        action: operators.Operator(T),
        input: []const T,
        coefficients: dense.ConstPanel(T),
        coefficient_counts: []const usize,
        output: dense.Panel(T),
    };
}
pub fn seriesSetWorkspaceLength(comptime T: type, action: operators.Operator(T), columns: usize) !usize {
    if (columns == 0) return 0;
    const states = std.math.mul(usize, action.rows, 3) catch return error.SizeOverflow;
    const results = std.math.mul(usize, action.rows, columns) catch return error.SizeOverflow;
    const vectors = std.math.add(usize, states, results) catch return error.SizeOverflow;
    const used = std.math.add(usize, vectors, action.workspace_elements) catch return error.SizeOverflow;
    _ = try storage.byteLength(used, @sizeOf(T));
    return used;
}
fn SeriesSetPreflight(comptime T: type) type {
    return struct { work: []T, maximum_count: usize };
}
fn seriesSetPreflight(comptime T: type, request: SeriesSetRequest(T), workspace: []T) !SeriesSetPreflight(T) {
    const columns = request.coefficient_counts.len;
    const coefficient_extent = try request.coefficients.check();
    const output_extent = try request.output.check();
    if (request.action.rows != request.action.columns or request.input.len != request.action.columns or request.output.shape.row_count != request.action.rows or request.output.shape.column_count != columns or request.coefficients.shape.column_count != columns) return error.DimensionMismatch;
    const used = try seriesSetWorkspaceLength(T, request.action, columns);
    if (workspace.len < used) return error.WorkspaceTooSmall;
    const work = workspace[0..used];
    const coefficients = request.coefficients.values[0..coefficient_extent];
    const output = request.output.values[0..output_extent];
    try noOverlap(T, request.input, work);
    try noOverlap(T, coefficients, work);
    try noOverlap(T, coefficients, output);
    try noOverlap(T, output, work);
    const count_range = try range(usize, request.coefficient_counts);
    if (storage.overlaps(count_range, try range(T, work)) or storage.overlaps(count_range, try range(T, output))) return error.AliasingNotAllowed;
    try checkFinite(T, request.input);
    var maximum: usize = 0;
    for (request.coefficient_counts, 0..) |count, column| {
        if (count == 0 or count > request.coefficients.shape.row_count) return error.InvalidOptions;
        maximum = @max(maximum, count);
        const values = request.coefficients.column(column);
        for (0..count) |order| if (!finite(T, values.at(order))) return error.NonFiniteInput;
    }
    return .{ .work = work, .maximum_count = maximum };
}
fn seriesSetControls(comptime T: type, operation: *const execution.context.Operation, request: SeriesSetRequest(T), maximum: usize) !void {
    const sync = request.action.synchronization;
    try synchronization.check(sync, operation, .{ .stage = .panel_columns, .control = @intCast(request.coefficient_counts.len) });
    try synchronization.check(sync, operation, .{ .stage = .max_order, .control = @intCast(maximum) });
    if (sync == null) return;
    for (request.coefficient_counts, 0..) |count, column| {
        try synchronization.check(sync, operation, .{ .stage = .coefficients_count, .index = @intCast(column), .control = @intCast(count) });
        const values = request.coefficients.column(column);
        for (0..count) |order| {
            const parts = synchronization.scalarParts(T, values.at(order));
            try synchronization.check(sync, operation, .{ .stage = .coefficient_real, .index = @intCast(order), .control = parts.real });
            if (comptime scalar.isComplex(T)) try synchronization.check(sync, operation, .{ .stage = .coefficient_imaginary, .index = @intCast(order), .control = parts.imaginary });
        }
    }
}
fn seriesSetInitial(comptime T: type, request: SeriesSetRequest(T), sum: []T, initial: []const T) !void {
    const rows = request.action.rows;
    for (0..request.coefficient_counts.len) |column| try initialSum(T, sum[column * rows ..][0..rows], initial, request.coefficients.column(column).at(0));
}
fn seriesSetArithmetic(comptime T: type, request: SeriesSetRequest(T), next: []T, previous: []const T, sum: []T, order: usize) !void {
    if (order > 1) try stepArithmetic(T, next, previous) else try checkProduced(T, next);
    const rows = request.action.rows;
    for (request.coefficient_counts, 0..) |count, column| {
        if (order >= count) continue;
        const coefficient = request.coefficients.column(column).at(order);
        for (sum[column * rows ..][0..rows], next) |*value, state| value.* = try linear(T, scalar.one(T), value.*, coefficient, state);
    }
}
/// Input may alias output: pack it before any action, then publish all logical
/// columns only after every action, arithmetic check and final agreement.
/// Coefficients/counts may not alias output; scratch may alias neither operand.
/// Scratch may change on failure. The callback borrows this one admission.
pub fn seriesSetAdmitted(comptime T: type, operation: *const execution.context.Operation, request: SeriesSetRequest(T), workspace: []T) !void {
    _ = operation.borrowedContext();
    const action = request.action;
    try synchronization.enter(T, action.synchronization, operation, .chebyshev_series_set);
    const prepared = try synchronization.resolve(action.synchronization, operation, .{ .stage = .preflight }, seriesSetPreflight(T, request, workspace));
    try seriesSetControls(T, operation, request, prepared.maximum_count);
    if (request.coefficient_counts.len == 0) return synchronization.check(action.synchronization, operation, .{ .stage = .commit });
    const rows = action.rows;
    const callback_work = prepared.work[0..action.workspace_elements];
    const vectors = prepared.work[action.workspace_elements..];
    var previous = vectors[0..rows];
    var current = vectors[rows..][0..rows];
    var next = vectors[2 * rows ..][0..rows];
    const sum = vectors[3 * rows ..];
    @memcpy(previous, request.input);
    try synchronization.resolve(action.synchronization, operation, .{ .stage = .local_arithmetic }, seriesSetInitial(T, request, sum, previous));
    for (1..prepared.maximum_count) |order| {
        try action.applyAdmitted(operation, if (order == 1) previous else current, next, callback_work);
        try synchronization.resolve(action.synchronization, operation, .{ .stage = .local_arithmetic, .index = @intCast(order) }, seriesSetArithmetic(T, request, next, previous, sum, order));
        if (order > 1) std.mem.swap([]T, &previous, &current);
        std.mem.swap([]T, &current, &next);
    }
    try synchronization.check(action.synchronization, operation, .{ .stage = .commit });
    for (0..request.coefficient_counts.len) |column| {
        const destination = request.output.column(column);
        for (0..rows) |row| destination.at(row).* = sum[column * rows + row];
    }
}

pub fn Managed(comptime Execution: type) type {
    return struct {
        const context = Execution.context;
        pub fn runSeriesSetBorrowedAdmitted(comptime T: type, operation: *const context.Operation, request: SeriesSetRequest(T), workspace: []T) !void {
            try seriesSetAdmitted(T, operation, request, workspace);
        }
        pub fn runSeriesSetScratchAdmitted(comptime T: type, operation: *const context.Operation, request: SeriesSetRequest(T)) !void {
            _ = operation.borrowedContext();
            try synchronization.enter(T, request.action.synchronization, operation, .chebyshev_series_set);
            const used = try synchronization.resolve(request.action.synchronization, operation, .{ .stage = .workspace }, seriesSetWorkspaceLength(T, request.action, request.coefficient_counts.len));
            var scratch = try synchronization.acquire(context, T, request.action.synchronization, operation, used, .vector_allocation);
            defer scratch.deinit();
            try seriesSetAdmitted(T, operation, request, scratch.lease.values);
        }
        pub fn runSeriesSetBorrowed(comptime T: type, owner: *context.Context, request: SeriesSetRequest(T), workspace: []T) !void {
            var operation = try synchronization.begin(request.action.synchronization, owner);
            defer operation.end();
            try seriesSetAdmitted(T, &operation, request, workspace);
        }
        pub fn runSeriesSetScratch(comptime T: type, owner: *context.Context, request: SeriesSetRequest(T)) !void {
            var operation = try synchronization.begin(request.action.synchronization, owner);
            defer operation.end();
            try runSeriesSetScratchAdmitted(T, &operation, request);
        }
        pub fn stepScratchAdmitted(comptime T: type, operation: *const context.Operation, action: operators.Operator(T), previous: []const T, current: []const T, next: []T) !void {
            _ = operation.borrowedContext();
            try synchronization.enter(T, action.synchronization, operation, .chebyshev_step);
            const used = try synchronization.resolve(action.synchronization, operation, .{ .stage = .workspace }, stepWorkspaceLength(T, action));
            var scratch = try synchronization.acquire(context, T, action.synchronization, operation, used, .vector_allocation);
            defer scratch.deinit();
            try stepAdmitted(T, operation, action, previous, current, next, scratch.lease.values);
        }
        pub fn runBorrowedAdmitted(comptime T: type, operation: *const context.Operation, request: Request(T), work: []T) !void {
            try seriesAdmitted(T, operation, request, work);
        }
        pub fn runScratchAdmitted(comptime T: type, operation: *const context.Operation, request: Request(T)) !void {
            _ = operation.borrowedContext();
            try synchronization.enter(T, request.action.synchronization, operation, .chebyshev_series);
            const used = try synchronization.resolve(request.action.synchronization, operation, .{ .stage = .workspace }, seriesWorkspaceLength(T, request.action));
            var scratch = try synchronization.acquire(context, T, request.action.synchronization, operation, used, .vector_allocation);
            defer scratch.deinit();
            try seriesAdmitted(T, operation, request, scratch.lease.values);
        }
        pub fn runPanelScratchAdmitted(comptime T: type, operation: *const context.Operation, request: PanelRequest(T)) !void {
            _ = operation.borrowedContext();
            try synchronization.enter(T, request.action.synchronization, operation, .chebyshev_panel);
            const used = try synchronization.resolve(request.action.synchronization, operation, .{ .stage = .workspace }, panelWorkspaceLength(T, request.action, request.input.shape.column_count));
            var scratch = try synchronization.acquire(context, T, request.action.synchronization, operation, used, .vector_allocation);
            defer scratch.deinit();
            try panelAdmitted(T, operation, request, scratch.lease.values);
        }
        pub fn runPanelScratch(comptime T: type, owner: *context.Context, request: PanelRequest(T)) !void {
            var operation = try synchronization.begin(request.action.synchronization, owner);
            defer operation.end();
            try runPanelScratchAdmitted(T, &operation, request);
        }
        pub fn runBorrowed(comptime T: type, owner: *context.Context, request: Request(T), work: []T) !void {
            var operation = try synchronization.begin(request.action.synchronization, owner);
            defer operation.end();
            try runBorrowedAdmitted(T, &operation, request, work);
        }
        pub fn runScratch(comptime T: type, owner: *context.Context, request: Request(T)) !void {
            var operation = try synchronization.begin(request.action.synchronization, owner);
            defer operation.end();
            try runScratchAdmitted(T, &operation, request);
        }
    };
}
