// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Spectral-bound-normalized Chebyshev filtering on the shared Operator.
//! Computes T_degree((H-center I)/half_width) X / T_degree((lower-center)/half_width).
//! Bounds are a caller policy; this is not an eigensolver or a spectrum proof.
//! Input/output may alias. Only a successful final agreement commits logical
//! output elements; scratch can change on failure. Callbacks borrow the same
//! admitted operation, and must finish their own communication before returning.
const std = @import("std");
const S = @import("../blas/core/shared/scalar.zig");
const A = @import("recurrence_arithmetic.zig");
const D = @import("dense.zig");
const O = @import("operator.zig");
const P = @import("synchronization.zig");
const storage = @import("storage.zig");
const execution = @import("zynum-execution");

pub fn Bounds(comptime T: type) type {
    return struct { lower: S.Real(T), cutoff: S.Real(T), upper: S.Real(T) };
}
pub fn Request(comptime T: type) type {
    return struct { action: O.Operator(T), input: []const T, output: []T, degree: usize, bounds: Bounds(T) };
}
pub fn PanelRequest(comptime T: type) type {
    return struct { action: O.Operator(T), input: D.ConstPanel(T), output: D.Panel(T), degree: usize, bounds: Bounds(T) };
}
fn Configuration(comptime T: type) type {
    return struct { center: A.Wide(T), half_width: A.Wide(T), denominator: A.Wide(T), sigma: A.Wide(T) };
}
fn configuration(comptime T: type, degree: usize, bounds: Bounds(T)) !Configuration(T) {
    if (degree == 0) return .{ .center = 0, .half_width = 1, .denominator = -1, .sigma = -1 };
    if (!std.math.isFinite(bounds.lower) or !std.math.isFinite(bounds.cutoff) or !std.math.isFinite(bounds.upper)) return error.NonFiniteInput;
    if (!(bounds.lower < bounds.cutoff and bounds.cutoff < bounds.upper)) return error.InvalidOptions;
    const W = A.Wide(T);
    const center = (@as(W, bounds.upper) + bounds.cutoff) / 2;
    const half_width = (@as(W, bounds.upper) - bounds.cutoff) / 2;
    const denominator = @as(W, bounds.lower) - center;
    return .{ .center = center, .half_width = half_width, .denominator = denominator, .sigma = half_width / denominator };
}
pub fn panelWorkspaceLength(comptime T: type, action: O.Operator(T), columns: usize) !usize {
    const elements = std.math.mul(usize, action.rows, columns) catch return error.SizeOverflow;
    const vectors = std.math.mul(usize, elements, 3) catch return error.SizeOverflow;
    const used = std.math.add(usize, vectors, try action.panelWorkspaceElements(S.one(T), columns)) catch return error.SizeOverflow;
    _ = try storage.byteLength(used, @sizeOf(T));
    return used;
}
pub fn workspaceLength(comptime T: type, action: O.Operator(T)) !usize {
    return panelWorkspaceLength(T, action, 1);
}
fn byteRange(comptime T: type, values: anytype) !storage.ByteRange {
    return storage.byteRange(@intFromPtr(values.ptr), try storage.byteLength(values.len, @sizeOf(T)));
}
fn pack(comptime T: type, values: []T, rows: usize, columns: usize) D.Panel(T) {
    return .{ .values = values, .shape = .{ .row_count = rows, .column_count = columns, .leading_dimension = rows, .layout = .column_major } };
}
fn preflight(comptime T: type, request: PanelRequest(T), workspace: []T) !Configuration(T) {
    const nx = try request.input.check();
    const ny = try request.output.check();
    if (request.action.rows != request.action.columns or request.input.shape.row_count != request.action.columns or request.output.shape.row_count != request.action.rows or request.input.shape.column_count != request.output.shape.column_count) return error.DimensionMismatch;
    const used = try panelWorkspaceLength(T, request.action, request.input.shape.column_count);
    if (workspace.len < used) return error.WorkspaceTooSmall;
    const scratch = try byteRange(T, workspace[0..used]);
    if (storage.overlaps(scratch, try byteRange(T, request.input.values[0..nx])) or storage.overlaps(scratch, try byteRange(T, request.output.values[0..ny]))) return error.AliasingNotAllowed;
    const config = try configuration(T, request.degree, request.bounds);
    for (0..request.input.shape.column_count) |column| {
        const x = request.input.column(column);
        for (0..request.action.rows) |row| if (!A.finite(T, x.at(row))) return error.NonFiniteInput;
    }
    return config;
}
fn arithmetic(comptime T: type, product: []T, current: []const T, previous: []const T, config: Configuration(T), scale: A.Wide(T), previous_factor: A.Wide(T)) !void {
    for (product, current, previous) |*hx, x, old| {
        if (!A.finite(T, hx.*)) return error.NonFiniteResult;
        hx.* = try A.shifted(T, hx.*, x, old, scale, config.center, previous_factor);
    }
}

pub fn panelAdmitted(comptime T: type, operation: *const execution.context.Operation, request: PanelRequest(T), workspace: []T) !void {
    _ = operation.borrowedContext();
    const action = request.action;
    try P.enter(T, action.synchronization, operation, .normalized_chebyshev);
    const config = try P.resolve(action.synchronization, operation, .{ .stage = .preflight }, preflight(T, request, workspace));
    try P.check(action.synchronization, operation, .{ .stage = .max_order, .control = @intCast(request.degree) });
    try P.check(action.synchronization, operation, .{ .stage = .panel_columns, .control = @intCast(request.input.shape.column_count) });
    if (request.degree > 0) try P.coefficients(S.Real(T), action.synchronization, operation, &.{ request.bounds.lower, request.bounds.cutoff, request.bounds.upper });
    const rows = action.rows;
    const columns = request.input.shape.column_count;
    const count = rows * columns;
    const callback_size = try action.panelWorkspaceElements(S.one(T), columns);
    const callback_work = workspace[0..callback_size];
    const vectors = workspace[callback_size..];
    var previous = vectors[0..count];
    var current = vectors[count..][0..count];
    var next = vectors[2 * count ..][0..count];
    for (0..columns) |column| {
        const x = request.input.column(column);
        for (0..rows) |row| previous[row + column * rows] = x.at(row);
    }
    if (request.degree > 0) {
        try action.applyPanelAdmitted(operation, S.one(T), pack(T, previous, rows, columns).asConst(), S.zero(T), pack(T, next, rows, columns), callback_work);
        // No previous term on the first step. Its zero value avoids reading
        // uninitialized current storage while preserving one common arithmetic.
        @memset(current, S.zero(T));
        try P.resolve(action.synchronization, operation, .{ .stage = .local_arithmetic, .index = 1 }, arithmetic(T, next, previous, current, config, 1 / config.denominator, 0));
        std.mem.swap([]T, &current, &next);
        var sigma = config.sigma;
        const gamma = 2 / config.sigma;
        for (1..request.degree) |order| {
            const sigma_next = 1 / (gamma - sigma);
            try action.applyPanelAdmitted(operation, S.one(T), pack(T, current, rows, columns).asConst(), S.zero(T), pack(T, next, rows, columns), callback_work);
            try P.resolve(action.synchronization, operation, .{ .stage = .local_arithmetic, .index = @intCast(order + 1) }, arithmetic(T, next, current, previous, config, 2 * sigma_next / config.half_width, sigma * sigma_next));
            sigma = sigma_next;
            std.mem.swap([]T, &previous, &current);
            std.mem.swap([]T, &current, &next);
        }
    }
    try P.check(action.synchronization, operation, .{ .stage = .commit });
    const result = if (request.degree == 0) previous else current;
    for (0..columns) |column| {
        const y = request.output.column(column);
        for (0..rows) |row| y.at(row).* = result[row + column * rows];
    }
}
pub fn runAdmitted(comptime T: type, operation: *const execution.context.Operation, request: Request(T), workspace: []T) !void {
    return panelAdmitted(T, operation, .{ .action = request.action, .input = .{ .values = request.input, .shape = .{ .row_count = request.input.len, .column_count = 1, .leading_dimension = request.input.len, .layout = .column_major } }, .output = pack(T, request.output, request.output.len, 1), .degree = request.degree, .bounds = request.bounds }, workspace);
}

pub fn Managed(comptime Execution: type) type {
    const C = Execution.context;
    return struct {
        pub fn runPanelBorrowedAdmitted(comptime T: type, operation: *const C.Operation, request: PanelRequest(T), workspace: []T) !void {
            return panelAdmitted(T, operation, request, workspace);
        }
        pub fn runPanelScratchAdmitted(comptime T: type, operation: *const C.Operation, request: PanelRequest(T)) !void {
            _ = operation.borrowedContext();
            try P.enter(T, request.action.synchronization, operation, .normalized_chebyshev);
            const used = try P.resolve(request.action.synchronization, operation, .{ .stage = .workspace }, panelWorkspaceLength(T, request.action, request.input.shape.column_count));
            var scratch = try P.acquire(C, T, request.action.synchronization, operation, used, .vector_allocation);
            defer scratch.deinit();
            return panelAdmitted(T, operation, request, scratch.lease.values);
        }
        pub fn runScratchAdmitted(comptime T: type, operation: *const C.Operation, request: Request(T)) !void {
            return runPanelScratchAdmitted(T, operation, .{ .action = request.action, .input = .{ .values = request.input, .shape = .{ .row_count = request.input.len, .column_count = 1, .leading_dimension = request.input.len, .layout = .column_major } }, .output = pack(T, request.output, request.output.len, 1), .degree = request.degree, .bounds = request.bounds });
        }
        pub fn runBorrowed(comptime T: type, owner: *C.Context, request: Request(T), workspace: []T) !void {
            var operation = try P.begin(request.action.synchronization, owner);
            defer operation.end();
            return runAdmitted(T, &operation, request, workspace);
        }
        pub fn runScratch(comptime T: type, owner: *C.Context, request: Request(T)) !void {
            var operation = try P.begin(request.action.synchronization, owner);
            defer operation.end();
            return runScratchAdmitted(T, &operation, request);
        }
        pub fn runPanelScratch(comptime T: type, owner: *C.Context, request: PanelRequest(T)) !void {
            var operation = try P.begin(request.action.synchronization, owner);
            defer operation.end();
            return runPanelScratchAdmitted(T, &operation, request);
        }
    };
}
