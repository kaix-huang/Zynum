// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//! Checked native invocation and one Context lifetime around the shared engine.
//! ABI controls/full advertised ranges and the acquired Guard are facade
//! preconditions; this module owns logical validation and numerical invocation.
const std = @import("std");
const execution = @import("zynum-execution");
pub const Engine = @import("checked_spectral.zig");
const A = @import("recurrence_arithmetic.zig");

pub const Kind = enum(u32) { series = 1, moments = 2 };
pub const Status = enum(i32) { ok, invalid_control, invalid_argument, view_invalid, overflow, aliasing, busy, context_failure, callback_failure, nonfinite_input, numerical_failure };
pub const Phase = enum(u32) { validation, admission, initialize, apply, recurrence, moment, commit, complete };
pub const Admission = enum { ok, closing, counter_overflow };
pub const Report = struct {
    phase: Phase = .validation,
    output_complete: bool = false,
    admission_attempted: bool = false,
    admission_succeeded: bool = false,
    admission: Admission = .ok,
    progress: Engine.Progress = .{},
};
/// Only Zig initializes/accesses this opaque object. Foreign callers supply
/// aligned raw storage, keep it stable, and never construct a foreign atomic.
/// Initialize exactly once before publication, exclusively and without copying.
pub const Guard = struct {
    active: std.atomic.Value(u32) = .init(0),
    pub fn acquire(self: *Guard) bool {
        return self.active.cmpxchgStrong(0, 1, .acquire, .monotonic) == null;
    }
    pub fn release(self: *Guard) void {
        self.active.store(0, .release);
    }
};
pub const ConstBuffer = struct { address: usize = 0, backing: u64 = 0, ld: u64 = 0 };
pub const Buffer = ConstBuffer;
pub const ConstPanel = struct {
    data: ?[*]const f64,
    rows: u64,
    columns: u64,
    ld: u64,
    pub fn cell(self: ConstPanel, i: u64, j: u64) *const f64 {
        return &self.data.?[@intCast(j * self.ld + i)];
    }
};
pub const Panel = struct {
    data: ?[*]f64,
    rows: u64,
    columns: u64,
    ld: u64,
    pub fn cell(self: Panel, i: u64, j: u64) *f64 {
        return &self.data.?[@intCast(j * self.ld + i)];
    }
    pub fn panel(self: Panel, start: u64, columns: u64) Panel {
        // Never form pointer offsets for a zero-length recurrence panel.
        return .{ .data = if (self.rows == 0 or columns == 0) null else self.data.? + @as(usize, @intCast(start * self.ld)), .rows = self.rows, .columns = columns, .ld = self.ld };
    }
};
pub const Apply = *const fn (?*anyopaque, *execution.Context, ConstPanel, Panel) i32;
pub const Operator = struct { apply: ?Apply, data: ?*anyopaque = null };
pub const Args = struct {
    kind: Kind,
    rows: u64,
    columns: u64,
    orders: u64,
    lower: f64,
    upper: f64,
    input: ConstBuffer,
    coefficients: ConstBuffer = .{},
    output: Buffer,
    work: Buffer,
    operator: Operator,
};
const SizeError = error{ InvalidArgument, Overflow, InvalidView };
pub const Layout = struct { panel_elements: u64, stage_offset: u64, work_elements: u64 };
fn host(value: u64) SizeError!u64 {
    _ = std.math.cast(usize, value) orelse return error.Overflow;
    return value;
}
fn mul(a: u64, b: u64) SizeError!u64 {
    return host(std.math.mul(u64, a, b) catch return error.Overflow);
}
fn add(a: u64, b: u64) SizeError!u64 {
    return host(std.math.add(u64, a, b) catch return error.Overflow);
}
fn bytes(elements: u64) SizeError!u64 {
    return mul(elements, @sizeOf(f64));
}
fn layout(kind: Kind, rows: u64, columns: u64, orders: u64) SizeError!Layout {
    if (orders == 0) return error.InvalidArgument;
    _ = try host(rows);
    _ = try host(columns);
    _ = try host(orders);
    _ = try mul(columns, if (kind == .series) 4 else 3);
    if (try bytes(rows) > std.math.maxInt(i64)) return error.Overflow;
    const order_bytes = try bytes(orders);
    if (kind == .moments and order_bytes > std.math.maxInt(i64)) return error.Overflow;
    const nw = try mul(rows, columns);
    _ = try bytes(nw);
    // All prefix offsets are checked even if another logical peer is empty.
    _ = try bytes(try mul(nw, 2));
    const stage_offset = try mul(nw, 3);
    _ = try bytes(stage_offset);
    const required = if (kind == .series) try mul(nw, 4) else blk: {
        const qw = try mul(orders, columns);
        _ = try bytes(qw);
        break :blk try add(stage_offset, qw);
    };
    _ = try bytes(required);
    return .{ .panel_elements = nw, .stage_offset = stage_offset, .work_elements = required };
}
pub fn workspaceElements(kind: Kind, rows: u64, columns: u64, orders: u64) SizeError!u64 {
    return (try layout(kind, rows, columns, orders)).work_elements;
}
fn extent(rows: u64, columns: u64, ld: u64) SizeError!u64 {
    if (rows == 0 or columns == 0) return 0;
    if (ld < rows) return error.InvalidView;
    const count = try add(try mul(columns - 1, ld), rows);
    _ = try bytes(count);
    return count;
}
fn sizeStatus(err: SizeError) Status {
    return switch (err) {
        error.InvalidArgument => .invalid_argument,
        error.Overflow => .overflow,
        error.InvalidView => .view_invalid,
    };
}
fn overlap(a: ConstBuffer, b: ConstBuffer) bool {
    if (a.backing == 0 or b.backing == 0) return false;
    // Full ranges have already passed the ABI preflight without overflow.
    return a.address < b.address + @as(usize, @intCast(b.backing * 8)) and b.address < a.address + @as(usize, @intCast(a.backing * 8));
}
fn constPanel(buffer: ConstBuffer, rows: u64, columns: u64) ConstPanel {
    return .{ .data = if (rows == 0 or columns == 0) null else @ptrFromInt(buffer.address), .rows = rows, .columns = columns, .ld = buffer.ld };
}
fn mutablePanel(buffer: Buffer, rows: u64, columns: u64) Panel {
    return .{ .data = if (rows == 0 or columns == 0) null else @ptrFromInt(buffer.address), .rows = rows, .columns = columns, .ld = buffer.ld };
}
const Callback = struct {
    context: *execution.Context,
    operator: Operator,
    pub fn apply(self: Callback, source: Panel, destination: Panel) i32 {
        return self.operator.apply.?(self.operator.data, self.context, .{ .data = source.data, .rows = source.rows, .columns = source.columns, .ld = source.ld }, destination);
    }
};
/// ABI preflight proved live controls/full ranges; caller holds the passive
/// guard. This owns one Context operation through final logical publication.
/// Neither admission nor these caller-thread loops claim a provider bulk charge.
pub fn invoke(context: *execution.Context, args: Args, report: *Report) Status {
    report.* = .{};
    if (args.operator.apply == null) return .invalid_argument;
    if (args.orders == 0) return .invalid_argument;
    if (!std.math.isFinite(args.lower)) return .invalid_argument;
    if (!std.math.isFinite(args.upper)) return .invalid_argument;
    if (!(args.lower < args.upper)) return .invalid_argument;
    const sizes = layout(args.kind, args.rows, args.columns, args.orders) catch |err| return sizeStatus(err);
    const input_required = extent(args.rows, args.columns, args.input.ld) catch |err| return sizeStatus(err);
    const output_rows = if (args.kind == .series) args.rows else args.orders;
    const output_required = extent(output_rows, args.columns, args.output.ld) catch |err| return sizeStatus(err);
    if (args.input.backing < input_required) return .view_invalid;
    if (args.output.backing < output_required) return .view_invalid;
    if (args.kind == .series and args.coefficients.backing < args.orders) return .view_invalid;
    if (args.work.backing < sizes.work_elements) return .view_invalid;
    if (args.input.address % @alignOf(f64) != 0) return .view_invalid;
    if (args.output.address % @alignOf(f64) != 0) return .view_invalid;
    if (args.kind == .series and args.coefficients.address % @alignOf(f64) != 0) return .view_invalid;
    if (args.work.address % @alignOf(f64) != 0) return .view_invalid;
    if (overlap(args.work, args.input) or overlap(args.work, args.output)) return .aliasing;
    if (args.kind == .series and (overlap(args.work, args.coefficients) or overlap(args.coefficients, args.output))) return .aliasing;
    const x = constPanel(args.input, args.rows, args.columns);
    const coefficients = if (args.kind == .series) constPanel(args.coefficients, args.orders, 1) else ConstPanel{ .data = null, .rows = 0, .columns = 0, .ld = 0 };
    if (args.kind == .series) for (0..@intCast(args.orders)) |k| {
        if (!std.math.isFinite(coefficients.cell(k, 0).*)) return .nonfinite_input;
    };
    // Preserve moments W0/N0 fast emptiness without hiding scalar validation.
    if (args.kind == .series or (args.rows != 0 and args.columns != 0)) for (0..@intCast(args.columns)) |j| for (0..@intCast(args.rows)) |i| {
        if (!std.math.isFinite(x.cell(i, j).*)) return .nonfinite_input;
    };
    report.phase = .admission;
    report.admission_attempted = true;
    var operation = context.begin() catch |err| {
        report.admission = switch (err) {
            error.Closing => .closing,
            error.CounterOverflow => .counter_overflow,
        };
        return .context_failure;
    };
    defer operation.end();
    report.admission_succeeded = true;
    const y = mutablePanel(args.output, output_rows, args.columns);
    const work = mutablePanel(.{ .address = args.work.address, .backing = args.work.backing, .ld = args.rows }, args.rows, (if (args.kind == .series) @as(u64, 4) else 3) * args.columns);
    const callback: Callback = .{ .context = context, .operator = args.operator };
    const result = if (args.kind == .series) Engine.series(A, callback, x, coefficients, y, work, args.rows, args.columns, args.orders, args.lower, args.upper, &report.progress) else blk: {
        const stage: Panel = .{ .data = if (args.columns == 0) null else @ptrFromInt(args.work.address + @as(usize, @intCast(sizes.stage_offset * 8))), .rows = args.orders, .columns = args.columns, .ld = args.orders };
        break :blk Engine.rawMoments(A, callback, x, y, work, stage, args.rows, args.columns, args.orders, args.lower, args.upper, &report.progress);
    };
    report.phase = switch (report.progress.phase) {
        .initialize => .initialize,
        .apply => .apply,
        .recurrence => .recurrence,
        .moment => .moment,
        .commit => .commit,
        .complete => .complete,
    };
    return switch (result) {
        .ok => blk: {
            report.output_complete = output_required != 0;
            break :blk .ok;
        },
        .callback_failure => .callback_failure,
        .numerical_failure => .numerical_failure,
    };
}
