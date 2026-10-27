// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//! Synchronous checked single-grid operation; all work uses its admitted provider.
const std = @import("std");
const execution = @import("zynum-execution");
const kernel = @import("periodic_laplacian.zig");
const storage = @import("storage.zig");

pub const PeriodicLaplacianF64 = struct {
    shape: [3]usize,
    input: []const f64,
    output: []f64,
    inv_h2: [3]f64,
};
pub const Error = execution.ExecutionError || error{ InvalidView, InvalidCoefficient, AliasingNotAllowed };

fn required(shape: [3]usize) Error!usize {
    for (shape) |length| if (length == 0) return 0;
    const plane = std.math.mul(usize, shape[0], shape[1]) catch return error.ExtentOverflow;
    const count = std.math.mul(usize, plane, shape[2]) catch return error.ExtentOverflow;
    _ = std.math.mul(usize, count, @sizeOf(f64)) catch return error.ExtentOverflow;
    return count;
}
fn range(values: []const f64) Error!storage.ByteRange {
    const address = @intFromPtr(values.ptr);
    if (address % @alignOf(f64) != 0 or (address == 0 and values.len != 0)) return error.InvalidView;
    const bytes = storage.byteLength(values.len, @sizeOf(f64)) catch return error.ExtentOverflow;
    return storage.byteRange(address, bytes) catch error.ExtentOverflow;
}
fn check(args: PeriodicLaplacianF64) Error!usize {
    const count = try required(args.shape);
    const input = try range(args.input);
    if (args.input.len < count) return error.InvalidView;
    const output = try range(args.output);
    if (args.output.len < count) return error.InvalidView;
    for (args.inv_h2) |h| if (!std.math.isFinite(h) or h <= 0) return error.InvalidCoefficient;
    if (storage.overlaps(input, output)) return error.AliasingNotAllowed;
    return count;
}
const Reader = struct {
    shape: [3]usize,
    values: []const f64,
    pub fn read(self: @This(), index: [3]usize) f64 {
        return self.values[(index[0] * self.shape[1] + index[1]) * self.shape[2] + index[2]];
    }
};
const Task = struct {
    args: PeriodicLaplacianF64,
    begin: usize,
    end: usize,
    fn run(raw: *const anyopaque, task_index: usize) void {
        const tasks: [*]const Task = @ptrCast(@alignCast(raw));
        const task = tasks[task_index];
        const shape = task.args.shape;
        const reader: Reader = .{ .shape = shape, .values = task.args.input };
        for (task.begin..task.end) |p| {
            const q = p / shape[2];
            const index: [3]usize = .{ q / shape[1], q % shape[1], p % shape[2] };
            task.args.output[p] = kernel.valueAt(shape, index, task.args.inv_h2, reader);
        }
    }
};
/// Borrow the one outer admission. Geometry and views have passed check().
fn applyAdmitted(operation: *const execution.Operation, args: PeriodicLaplacianF64, count: usize, report: *execution.CallReport) Error!void {
    const resources = execution.admittedResources(operation);
    const limit = @min(resources.maxThreads(), execution.max_tasks);
    if (limit == 0) return error.Unsupported;
    if (count == 0) return;
    const task_count = @min(count, limit);
    var tasks = try resources.acquireScratch(Task, task_count, report);
    defer tasks.release(report);
    const quotient = count / task_count;
    const remainder = count % task_count;
    var begin: usize = 0;
    for (tasks.values, 0..) |*task, index| {
        const length = quotient + @intFromBool(index < remainder);
        const end = begin + length;
        task.* = .{ .args = args, .begin = begin, .end = end };
        begin = end;
    }
    try resources.runChecked(Task.run, @ptrCast(tasks.values.ptr), tasks.values.len, report);
}
/// No finite-output promise: success means completed original IEEE arithmetic.
/// Caller buffers, Context, report and implementation remain live through return.
pub fn periodicLaplacianF64(context: *execution.Context, args: PeriodicLaplacianF64, report: *execution.CallReport) Error!void {
    report.* = .{};
    const count = try check(args);
    var operation = context.begin() catch |err| return switch (err) {
        error.Closing => error.ContextClosing,
        error.CounterOverflow => error.CounterOverflow,
    };
    defer operation.end();
    try applyAdmitted(&operation, args, count, report);
}
