// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//! Local original-Heteris numerical/ownership evidence; no cluster qualification.
const std = @import("std");
const native = @import("zynum-core").native;
const execution = @import("zynum-execution");
const c = @import("heteris-provider-c");
const Mode = enum { normal, budget, oom_second, reject, callback_error };
var mode: Mode = .normal;
var original: c.heteris_runtime_provider_api_v1 = undefined;
var allocations: usize = 0;
var submissions: usize = 0;
var callbacks: std.atomic.Value(usize) = .init(0);
var caller: std.Thread.Id = undefined;
fn allocate(token: c.heteris_rp_token, bytes: u64, alignment: u64, out: [*c]c.heteris_rp_scratch_v1) callconv(.c) i32 {
    allocations += 1;
    if (mode == .budget) return c.HETERIS_BUDGET_EXCEEDED;
    if (mode == .oom_second and allocations == 2) return c.HETERIS_OUT_OF_MEMORY;
    return original.scratch_allocate.?(token, bytes, alignment, out);
}
const Forward = struct { callback: c.heteris_rp_cpu_range_fn, data: ?*anyopaque };
fn callback(data: ?*anyopaque, begin: u64, end: u64) callconv(.c) i32 {
    if (std.Thread.getCurrentId() == caller) @panic("mathematics ran on submitting caller");
    const forward: *const Forward = @ptrCast(@alignCast(data.?));
    _ = callbacks.fetchAdd(1, .monotonic);
    const status = forward.callback.?(forward.data, begin, end);
    return if (status == 0 and mode == .callback_error) -1903 else status;
}
fn submit(token: c.heteris_rp_token, begin: u64, end: u64, grain: u64, limit: u32, task_fn: c.heteris_rp_cpu_range_fn, data: ?*anyopaque, out: [*c]c.heteris_rp_bulk_result_v1) callconv(.c) i32 {
    submissions += 1;
    if (mode == .reject) {
        out[0] = .{ .abi_version = 1, .struct_size = @sizeOf(c.heteris_rp_bulk_result_v1), .state = c.HETERIS_RP_REJECTED, .reserved = 0, .execution_status = -778, .callback_status = 0, .callbacks_started = 0, .callbacks_completed = 0 };
        return -778;
    }
    var forward: Forward = .{ .callback = task_fn, .data = data };
    return original.cpu_bulk_sync.?(token, begin, end, grain, limit, callback, &forward, out);
}
const Fixture = struct {
    root: c.heteris_rp_token = undefined,
    source: c.heteris_runtime_provider_v1 = undefined,
    api: c.heteris_runtime_provider_api_v1 = undefined,
    context: execution.Context = undefined,
    report: execution.CallReport = .{},
    fn init(self: *@This(), limit: u32) !void {
        mode = .normal;
        allocations = 0;
        submissions = 0;
        callbacks.store(0, .monotonic);
        caller = std.Thread.getCurrentId();
        const desc: c.heteris_rp_cpu_create_desc_v1 = .{ .abi_version = 1, .struct_size = @sizeOf(c.heteris_rp_cpu_create_desc_v1), .compute_workers = 2, .managed_thread_limit = 3, .max_tasks = 16, .max_dependencies = 8, .host_bytes_limit = 65536, .task_scratch_bytes_limit = 0, .reserved = .{ 0, 0 } };
        try std.testing.expectEqual(@as(i32, 0), c.heteris_cpu_runtime_create_v1(&desc, &self.root));
        try std.testing.expectEqual(@as(i32, 0), c.heteris_cpu_runtime_export_provider_v1(self.root, &self.source));
        original = self.source.api[0];
        self.api = original;
        self.api.scratch_allocate = allocate;
        self.api.cpu_bulk_sync = submit;
        self.source.api = &self.api;
        self.context = try execution.Context.initIntegrated(&self.source, .{ .max_parallel_tasks = limit }, &self.report);
    }
    fn deinit(self: *@This()) void {
        _ = self.context.requestClose();
        if (self.context.tryProgressClose(&self.report) != .closed) @panic("native math leaked admission or scratch");
        if (original.provider_release.?(self.source.provider) != 0) @panic("provider reference leak");
        if (c.heteris_cpu_runtime_destroy_v1(self.root) != 0) @panic("runtime resource leak");
    }
};
fn args(a: []const f64, b: []const f64, output: []f64, m: i32, n: i32, k: i32, lda: i32, ldb: i32, ldc: i32, alpha: f64, beta: f64) native.MatrixProductF64 {
    return .{
        .left_matrix = .{ .values = a, .row_count = m, .column_count = k, .leading_dimension = lda },
        .right_matrix = .{ .values = b, .row_count = k, .column_count = n, .leading_dimension = ldb },
        .result_matrix = .{ .values = output, .row_count = m, .column_count = n, .leading_dimension = ldc },
        .product_scale = alpha,
        .result_scale = beta,
    };
}

test "native original generic GEMM covers padded row-column partitions and single-task execution" {
    const shapes = [_][3]usize{ .{ 1, 1, 1 }, .{ 1, 9, 19 }, .{ 9, 1, 19 }, .{ 7, 9, 5 }, .{ 33, 17, 19 }, .{ 17, 2048, 128 }, .{ 34, 2048, 256 } };
    for ([_]u32{ 1, 2 }) |limit| {
        var fixture: Fixture = .{};
        try fixture.init(limit);
        defer fixture.deinit();
        for (shapes) |shape| {
            const m, const n, const k = shape;
            const lda = m + 2;
            const ldb = k + 3;
            const ldc = m + 4;
            const a = try std.testing.allocator.alloc(f64, lda * k);
            defer std.testing.allocator.free(a);
            const b = try std.testing.allocator.alloc(f64, ldb * n);
            defer std.testing.allocator.free(b);
            const output = try std.testing.allocator.alloc(f64, ldc * n);
            defer std.testing.allocator.free(output);
            for (a, 0..) |*value, i| value.* = @as(f64, @floatFromInt(i % 7)) - 3;
            for (b, 0..) |*value, i| value.* = @as(f64, @floatFromInt(i % 5)) - 2;
            @memset(output, 7);
            try native.matrixMultiplyF64NN(&fixture.context, args(a, b, output, @intCast(m), @intCast(n), @intCast(k), @intCast(lda), @intCast(ldb), @intCast(ldc), 1.5, -0.25), &fixture.report);
            try std.testing.expectEqual(execution.OutputState.complete, fixture.report.output);
            for (0..n) |j| {
                for (0..m) |i| {
                    var sum: f64 = 0;
                    for (0..k) |q| sum += a[i + q * lda] * b[q + j * ldb];
                    try std.testing.expectEqual(1.5 * sum - 0.25 * 7, output[i + j * ldc]);
                }
                for (m..ldc) |i| try std.testing.expectEqual(@as(f64, 7), output[i + j * ldc]);
            }
        }
        try std.testing.expectEqual(@as(usize, shapes.len), submissions);
        try std.testing.expectEqual(@as(usize, shapes.len * 2), allocations);
        try std.testing.expect(callbacks.load(.monotonic) >= shapes.len);
    }
}

test "scaling-only and beta-zero paths use provider and do not read empty operands or NaN output" {
    var fixture: Fixture = .{};
    try fixture.init(2);
    defer fixture.deinit();
    var output = [_]f64{ std.math.nan(f64), std.math.nan(f64), 99, std.math.nan(f64), std.math.nan(f64), 99 };
    try native.matrixMultiplyF64NN(&fixture.context, args(&.{}, &.{}, &output, 2, 2, 0, 2, 0, 3, 1, 0), &fixture.report);
    try std.testing.expectEqualSlices(f64, &.{ 0, 0, 99, 0, 0, 99 }, &output);
    const nan: [4]f64 = @splat(std.math.nan(f64));
    output = .{ 2, 3, 99, 4, 5, 99 };
    try native.matrixMultiplyF64NN(&fixture.context, args(&nan, &nan, &output, 2, 2, 2, 2, 2, 3, 0, 2), &fixture.report);
    try std.testing.expectEqualSlices(f64, &.{ 4, 6, 99, 8, 10, 99 }, &output);
    try std.testing.expectEqual(@as(usize, 2), submissions);
    try std.testing.expectEqual(@as(usize, 4), allocations);
    var singleton = [_]f64{std.math.nan(f64)};
    try native.matrixMultiplyF64NN(&fixture.context, args(&.{2}, &.{3}, &singleton, 1, 1, 1, 1, 1, 1, 1, 0), &fixture.report);
    try std.testing.expectEqual(@as(f64, 6), singleton[0]);
}

test "scratch failures and rejected submissions leave output unchanged without replay" {
    for ([_]Mode{ .budget, .oom_second, .reject }) |failure| {
        var fixture: Fixture = .{};
        try fixture.init(2);
        defer fixture.deinit();
        mode = failure;
        var output = [_]f64{ 7, 7, 7, 7 };
        const input = [_]f64{ 1, 2, 3, 4 };
        const expected: native.NativeError = switch (failure) {
            .budget => error.ResourceBudgetExceeded,
            .oom_second => error.OutOfMemory,
            .reject => error.ProviderRejected,
            else => unreachable,
        };
        try std.testing.expectError(expected, native.matrixMultiplyF64NN(&fixture.context, args(&input, &input, &output, 2, 2, 2, 2, 2, 2, 1, 1), &fixture.report));
        try std.testing.expectEqualSlices(f64, &.{ 7, 7, 7, 7 }, &output);
        try std.testing.expectEqual(execution.OutputState.unchanged, fixture.report.output);
        try std.testing.expectEqual(@as(usize, if (failure == .reject) 1 else 0), submissions);
        try std.testing.expectEqual(@as(usize, 0), callbacks.load(.monotonic));
        // A later independent request succeeds on the same Runtime after all
        // failed-call leases are freed. It is not an automatic retry.
        mode = .normal;
        try native.matrixMultiplyF64NN(&fixture.context, args(&input, &input, &output, 2, 2, 2, 2, 2, 2, 1, 1), &fixture.report);
        try std.testing.expectEqualSlices(f64, &.{ 14, 17, 22, 29 }, &output);
    }
}

test "real-worker callback failure is quiescent partial-invalid and never replays completed tiles" {
    var fixture: Fixture = .{};
    try fixture.init(2);
    defer fixture.deinit();
    mode = .callback_error;
    var output = [_]f64{ 7, 7, 7, 7 };
    const input = [_]f64{ 1, 2, 3, 4 };
    try std.testing.expectError(error.ProviderCallbackFailed, native.matrixMultiplyF64NN(&fixture.context, args(&input, &input, &output, 2, 2, 2, 2, 2, 2, 1, 1), &fixture.report));
    try std.testing.expectEqual(execution.OutputState.partial_or_invalid, fixture.report.output);
    try std.testing.expectEqual(execution.SubmissionState.quiescent, fixture.report.provider.state);
    try std.testing.expectEqual(@as(i32, -1903), fixture.report.provider.callback_status);
    try std.testing.expectEqual(@as(usize, 1), submissions);
    const expected = [_]f64{ 14, 17, 22, 29 };
    for (output, expected) |value, full| try std.testing.expect(value == 7 or value == full);
    try std.testing.expect(callbacks.load(.monotonic) > 0);
}

test "invalid and empty views never allocate or submit; closing Context rejects valid work" {
    var fixture: Fixture = .{};
    try fixture.init(1);
    defer fixture.deinit();
    var output = [_]f64{ 7, 7, 7, 7 };
    const input = [_]f64{ 1, 2, 3, 4 };
    var request = args(&input, &input, &output, 2, 2, 2, 2, 2, 2, 1, 0);
    request.left_matrix.values = input[0..3];
    try std.testing.expectError(error.BufferTooSmall, native.matrixMultiplyF64NN(&fixture.context, request, &fixture.report));
    request.left_matrix.values = &output;
    try std.testing.expectError(error.AliasingNotAllowed, native.matrixMultiplyF64NN(&fixture.context, request, &fixture.report));
    request.left_matrix.values = &input;
    request.left_matrix.operation = .transposed;
    try std.testing.expectError(error.Unsupported, native.matrixMultiplyF64NN(&fixture.context, request, &fixture.report));
    request.left_matrix.operation = .normal;
    request.right_matrix.row_count = 1;
    try std.testing.expectError(error.DimensionMismatch, native.matrixMultiplyF64NN(&fixture.context, request, &fixture.report));
    try native.matrixMultiplyF64NN(&fixture.context, args(&.{}, &input, output[0..0], 0, 2, 2, 0, 2, 0, 1, 0), &fixture.report);
    try std.testing.expectEqual(execution.OutputState.unchanged, fixture.report.output);
    try std.testing.expectEqual(@as(usize, 0), allocations);
    try std.testing.expectEqual(@as(usize, 0), submissions);
    _ = fixture.context.requestClose();
    try std.testing.expectError(error.ContextClosing, native.matrixMultiplyF64NN(&fixture.context, args(&input, &input, &output, 2, 2, 2, 2, 2, 2, 1, 0), &fixture.report));
    try std.testing.expectEqualSlices(f64, &.{ 7, 7, 7, 7 }, &output);
}
