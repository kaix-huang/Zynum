// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Shared weighted Hermitian subspace projection over native Dense panels.
//! H=Y* W Z and S=Y* W Y use caller-supplied Y/Z and optional nonnegative
//! local weights. One optional bulk reduction sees packed column-major H,S.
//! Both reduced matrices are Hermitianized and commit together. The caller
//! decides positive definiteness, eigensolver choice and result usability.
const std = @import("std");
const dense = @import("dense.zig");
const storage = @import("storage.zig");
const scalar = @import("../blas/core/shared/scalar.zig");
const reductions = @import("reductions.zig");
const arithmetic = @import("recurrence_arithmetic.zig");
const execution = @import("zynum-execution");
const phase = @import("synchronization.zig");

pub fn Reduction(comptime T: type) type {
    return struct {
        context: *const anyopaque,
        reduce_fn: *const fn (*const anyopaque, *const execution.context.Operation, []T) anyerror!void,
    };
}
pub fn Request(comptime T: type) type {
    return struct {
        basis: dense.ConstPanel(T),
        applied: dense.ConstPanel(T),
        weights: ?[]const scalar.Real(T) = null,
        hamiltonian: dense.Panel(T),
        overlap: dense.Panel(T),
        reduction: ?Reduction(T) = null,
        synchronization: ?phase.Synchronization = null,
    };
}
/// T elements: three packed local panels plus two projected square matrices.
/// Empty column sets are validated no-ops and use no workspace/reduction.
pub fn workspaceLength(comptime T: type, rows: usize, columns: usize) !usize {
    if (columns == 0) return 0;
    const panel = std.math.mul(usize, rows, columns) catch return error.SizeOverflow;
    const square = std.math.mul(usize, columns, columns) catch return error.SizeOverflow;
    const panels = std.math.mul(usize, panel, 3) catch return error.SizeOverflow;
    const results = std.math.mul(usize, square, 2) catch return error.SizeOverflow;
    const total = std.math.add(usize, panels, results) catch return error.SizeOverflow;
    _ = try storage.byteLength(total, @sizeOf(T));
    return total;
}
fn range(values: anytype) !storage.ByteRange {
    return storage.byteRange(@intFromPtr(values.ptr), try storage.byteLength(values.len, @sizeOf(@typeInfo(@TypeOf(values)).pointer.child)));
}
fn separate(a: anytype, b: anytype) !void {
    if (storage.overlaps(try range(a), try range(b))) return error.AliasingNotAllowed;
}
fn preflight(comptime T: type, r: Request(T), workspace: []T) ![]T {
    const rows = r.basis.shape.row_count;
    const columns = r.basis.shape.column_count;
    const y_used = try r.basis.check();
    const z_used = try r.applied.check();
    const h_used = try r.hamiltonian.check();
    const s_used = try r.overlap.check();
    if (r.applied.shape.row_count != rows or r.applied.shape.column_count != columns or
        r.hamiltonian.shape.row_count != columns or r.hamiltonian.shape.column_count != columns or
        r.overlap.shape.row_count != columns or r.overlap.shape.column_count != columns)
        return error.DimensionMismatch;
    if (r.weights) |weights| if (weights.len != rows) return error.DimensionMismatch;
    const used = try workspaceLength(T, rows, columns);
    if (workspace.len < used) return error.WorkspaceTooSmall;
    const work = workspace[0..used];
    const y = r.basis.values[0..y_used];
    const z = r.applied.values[0..z_used];
    const h = r.hamiltonian.values[0..h_used];
    const s = r.overlap.values[0..s_used];
    try separate(h, s);
    for ([_][]const T{ y, z, h, s }) |values| try separate(values, work);
    for ([_][]const T{ h, s }) |out| {
        try separate(out, y);
        try separate(out, z);
    }
    if (r.weights) |weights| {
        try separate(weights, work);
        try separate(weights, h);
        try separate(weights, s);
        if (columns != 0) for (weights) |weight| {
            if (!std.math.isFinite(weight)) return error.NonFiniteInput;
            if (weight < 0) return error.InvalidMetric;
        };
    }
    for (0..columns) |column| for (0..rows) |row| {
        if (!reductions.finite(T, r.basis.column(column).at(row)) or
            !reductions.finite(T, r.applied.column(column).at(row))) return error.NonFiniteInput;
    };
    return work;
}
fn conjugate(comptime T: type, x: T) T {
    return if (comptime scalar.isComplex(T)) .{ .re = x.re, .im = -x.im } else x;
}
fn local(comptime T: type, r: Request(T), work: []T) ![]T {
    const rows = r.basis.shape.row_count;
    const columns = r.basis.shape.column_count;
    const count = rows * columns;
    const square = columns * columns;
    const y = work[0..count];
    const wy = work[count .. 2 * count];
    const wz = work[2 * count .. 3 * count];
    const projected = work[3 * count ..][0 .. 2 * square];
    for (0..columns) |column| for (0..rows) |row| {
        const value = r.basis.column(column).at(row);
        const action = r.applied.column(column).at(row);
        const weight = scalar.realScalar(T, if (r.weights) |weights| weights[row] else 1);
        const offset = column * rows + row;
        y[offset] = value;
        wy[offset] = try arithmetic.linear(T, weight, value, scalar.zero(T), scalar.zero(T));
        wz[offset] = try arithmetic.linear(T, weight, action, scalar.zero(T), scalar.zero(T));
    };
    for (0..columns) |column| for (0..columns) |row| {
        const left = y[row * rows ..][0..rows];
        projected[row + column * columns] = try reductions.localInner(T, left, wz[column * rows ..][0..rows]);
        projected[square + row + column * columns] = try reductions.localInner(T, left, wy[column * rows ..][0..rows]);
    };
    return projected;
}
fn reduce(comptime T: type, operation: *const execution.context.Operation, r: Request(T), values: []T) !void {
    if (r.reduction) |reduction| try reduction.reduce_fn(reduction.context, operation, values);
    for (values) |value| if (!reductions.finite(T, value)) return error.NonFiniteReduction;
}
fn hermitianize(comptime T: type, values: []T, columns: usize) !void {
    const half = scalar.realScalar(T, 0.5);
    for (0..2) |matrix| {
        const data = values[matrix * columns * columns ..][0 .. columns * columns];
        for (0..columns) |column| {
            if (comptime scalar.isComplex(T)) data[column + column * columns].im = 0;
            for (column + 1..columns) |row| {
                const value = try arithmetic.linear(T, half, data[row + column * columns], half, conjugate(T, data[column + row * columns]));
                data[row + column * columns] = value;
                data[column + row * columns] = conjugate(T, value);
            }
        }
    }
}
pub fn runAdmitted(comptime T: type, operation: *const execution.context.Operation, r: Request(T), workspace: []T) !void {
    _ = operation.borrowedContext();
    try phase.enter(T, r.synchronization, operation, .subspace_projection);
    const work = try phase.resolve(r.synchronization, operation, .{ .stage = .preflight }, preflight(T, r, workspace));
    const columns = r.basis.shape.column_count;
    try phase.check(r.synchronization, operation, .{ .stage = .panel_columns, .control = @intCast(columns) });
    try phase.check(r.synchronization, operation, .{ .stage = .panel_backend, .control = @intFromBool(r.weights != null) });
    try phase.check(r.synchronization, operation, .{ .stage = .reduction_backend, .control = @intFromBool(r.reduction != null) });
    if (columns == 0) return phase.check(r.synchronization, operation, .{ .stage = .commit });
    const values = try phase.resolve(r.synchronization, operation, .{ .stage = .local_arithmetic }, local(T, r, work));
    try phase.resolve(r.synchronization, operation, .{ .stage = .projection_real }, reduce(T, operation, r, values));
    try phase.resolve(r.synchronization, operation, .{ .stage = .local_arithmetic, .index = 1 }, hermitianize(T, values, columns));
    try phase.check(r.synchronization, operation, .{ .stage = .commit });
    for (0..columns) |column| for (0..columns) |row| {
        r.hamiltonian.column(column).at(row).* = values[row + column * columns];
        r.overlap.column(column).at(row).* = values[columns * columns + row + column * columns];
    };
}
pub fn Managed(comptime E: type) type {
    return struct {
        const context = E.context;
        pub fn runBorrowedAdmitted(comptime T: type, operation: *const context.Operation, request: Request(T), workspace: []T) !void {
            try runAdmitted(T, operation, request, workspace);
        }
        pub fn runScratchAdmitted(comptime T: type, operation: *const context.Operation, request: Request(T)) !void {
            _ = operation.borrowedContext();
            try phase.enter(T, request.synchronization, operation, .subspace_projection);
            const used = try phase.resolve(request.synchronization, operation, .{ .stage = .workspace }, workspaceLength(T, request.basis.shape.row_count, request.basis.shape.column_count));
            var work = try phase.acquire(context, T, request.synchronization, operation, used, .vector_allocation);
            defer work.deinit();
            try runAdmitted(T, operation, request, work.lease.values);
        }
        pub fn runBorrowed(comptime T: type, owner: *context.Context, request: Request(T), workspace: []T) !void {
            var operation = try phase.begin(request.synchronization, owner);
            defer operation.end();
            try runAdmitted(T, &operation, request, workspace);
        }
        pub fn runScratch(comptime T: type, owner: *context.Context, request: Request(T)) !void {
            var operation = try phase.begin(request.synchronization, owner);
            defer operation.end();
            try runScratchAdmitted(T, &operation, request);
        }
    };
}
