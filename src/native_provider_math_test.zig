// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//! Local original-Heteris numerical/ownership evidence; no cluster qualification.
const std = @import("std");
const native = @import("zynum-core").native;
const execution = @import("zynum-execution");
const c = @import("heteris-provider-c");
const tn_reference = @import("gemm-test-reference").referenceGemmTransposedA;
const Mode = enum { normal, budget, oom_second, reject, callback_error };
var mode: Mode = .normal;
var original: c.heteris_runtime_provider_api_v1 = undefined;
var allocations: usize = 0;
var submissions: usize = 0;
var callbacks: std.atomic.Value(usize) = .init(0);
var caller: std.Thread.Id = undefined;
const BulkFailure = enum { none, reject, execution, callback };
var fail_allocation: usize = 0;
var allocation_status: i32 = 0;
var fail_submission: usize = 0;
var bulk_failure: BulkFailure = .none;
var live_scratch: usize = 0;
var bulk_results: [2]c.heteris_rp_bulk_result_v1 = undefined;
var bulk_statuses: [2]i32 = undefined;
var close_after_packing: ?*execution.Context = null;
var close_was_busy: bool = false;
var second_phase_admitted: bool = false;
var grid_close_during: ?*execution.Context = null;
var grid_close_during_busy: std.atomic.Value(bool) = .init(false);
fn allocate(token: c.heteris_rp_token, bytes: u64, alignment: u64, out: [*c]c.heteris_rp_scratch_v1) callconv(.c) i32 {
    allocations += 1;
    if (mode == .budget) return c.HETERIS_BUDGET_EXCEEDED;
    if (mode == .oom_second and allocations == 2) return c.HETERIS_OUT_OF_MEMORY;
    if (allocations == fail_allocation) return allocation_status;
    const status = original.scratch_allocate.?(token, bytes, alignment, out);
    if (status == 0) live_scratch += 1;
    return status;
}
fn releaseScratch(token: c.heteris_rp_token) callconv(.c) i32 {
    const status = original.scratch_release.?(token);
    if (status == 0) live_scratch -= 1;
    return status;
}
const Forward = struct { callback: c.heteris_rp_cpu_range_fn, data: ?*anyopaque, submission: usize };
fn callback(data: ?*anyopaque, begin: u64, end: u64) callconv(.c) i32 {
    if (std.Thread.getCurrentId() == caller) @panic("mathematics ran on submitting caller");
    const forward: *const Forward = @ptrCast(@alignCast(data.?));
    _ = callbacks.fetchAdd(1, .monotonic);
    if (grid_close_during) |context| {
        _ = context.requestClose();
        var report: execution.CallReport = .{};
        if (context.tryProgressClose(&report) == .busy and context.snapshot().active_operations == 1)
            grid_close_during_busy.store(true, .release);
    }
    const status = forward.callback.?(forward.data, begin, end);
    return if (status == 0 and (mode == .callback_error or (forward.submission == fail_submission and bulk_failure == .callback))) -1903 else status;
}
fn submit(token: c.heteris_rp_token, begin: u64, end: u64, grain: u64, limit: u32, task_fn: c.heteris_rp_cpu_range_fn, data: ?*anyopaque, out: [*c]c.heteris_rp_bulk_result_v1) callconv(.c) i32 {
    submissions += 1;
    if (mode == .reject or (submissions == fail_submission and bulk_failure == .reject)) {
        out[0] = .{ .abi_version = 1, .struct_size = @sizeOf(c.heteris_rp_bulk_result_v1), .state = c.HETERIS_RP_REJECTED, .reserved = 0, .execution_status = -778, .callback_status = 0, .callbacks_started = 0, .callbacks_completed = 0 };
        if (submissions <= 2) {
            bulk_results[submissions - 1] = out[0];
            bulk_statuses[submissions - 1] = -778;
        }
        return -778;
    }
    if (submissions == 2) if (close_after_packing) |context| {
        const snapshot = context.snapshot();
        second_phase_admitted = snapshot.active_operations == 1 and snapshot.resource_pins == 3;
    };
    var forward: Forward = .{ .callback = task_fn, .data = data, .submission = submissions };
    var status = original.cpu_bulk_sync.?(token, begin, end, grain, limit, callback, &forward, out);
    if (submissions == fail_submission and bulk_failure == .execution) {
        out[0].execution_status = -1904;
        status = -1904;
    }
    if (submissions <= 2) {
        bulk_results[submissions - 1] = out[0];
        bulk_statuses[submissions - 1] = status;
    }
    if (submissions == 1) if (close_after_packing) |context| {
        _ = context.requestClose();
        var report: execution.CallReport = .{};
        close_was_busy = context.tryProgressClose(&report) == .busy and context.snapshot().active_operations == 1;
    };
    return status;
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
        fail_allocation = 0;
        allocation_status = 0;
        fail_submission = 0;
        bulk_failure = .none;
        live_scratch = 0;
        close_after_packing = null;
        close_was_busy = false;
        second_phase_admitted = false;
        grid_close_during = null;
        grid_close_during_busy.store(false, .monotonic);
        const desc: c.heteris_rp_cpu_create_desc_v1 = .{ .abi_version = 1, .struct_size = @sizeOf(c.heteris_rp_cpu_create_desc_v1), .compute_workers = 2, .managed_thread_limit = 3, .max_tasks = 16, .max_dependencies = 8, .host_bytes_limit = 4 * 1024 * 1024, .task_scratch_bytes_limit = 0, .reserved = .{ 0, 0 } };
        try std.testing.expectEqual(@as(i32, 0), c.heteris_cpu_runtime_create_v1(&desc, &self.root));
        try std.testing.expectEqual(@as(i32, 0), c.heteris_cpu_runtime_export_provider_v1(self.root, &self.source));
        original = self.source.api[0];
        self.api = original;
        self.api.scratch_allocate = allocate;
        self.api.scratch_release = releaseScratch;
        self.api.cpu_bulk_sync = submit;
        self.source.api = &self.api;
        self.context = try execution.Context.initIntegrated(&self.source, .{ .max_parallel_tasks = limit }, &self.report);
    }
    fn deinit(self: *@This()) void {
        if (live_scratch != 0) @panic("native math scratch leak");
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

fn tnArgs(a: []const f64, b: []const f64, output: []f64, m: i32, n: i32, k: i32, lda: i32, ldb: i32, ldc: i32, alpha: f64, beta: f64) native.MatrixProductF64 {
    var result = args(a, b, output, m, n, k, lda, ldb, ldc, alpha, beta);
    result.left_matrix.row_count = k;
    result.left_matrix.column_count = m;
    result.left_matrix.operation = .transposed;
    return result;
}
fn expectBulkDiagnostic(report: execution.CallReport, phase: usize) !void {
    const result = bulk_results[phase - 1];
    try std.testing.expectEqual(execution.ResourceStage.submit, report.provider.stage);
    try std.testing.expectEqual(bulk_statuses[phase - 1], report.provider.returned_status);
    try std.testing.expectEqual(result.execution_status, report.provider.execution_status);
    try std.testing.expectEqual(result.callback_status, report.provider.callback_status);
    try std.testing.expectEqual(@as(i32, 0), report.provider.cleanup_status);
    try std.testing.expectEqual(result.callbacks_started, report.provider.callbacks_started);
    try std.testing.expectEqual(result.callbacks_completed, report.provider.callbacks_completed);
    try std.testing.expectEqual(if (result.state == c.HETERIS_RP_REJECTED) execution.SubmissionState.rejected else execution.SubmissionState.quiescent, report.provider.state);
}
fn sampleTN(row: usize, column: usize, salt: usize) f64 {
    const raw = ((row + 1) * (salt + 3) + (column + 1) * (salt + 7)) % 29;
    return (@as(f64, @floatFromInt(raw)) - 14.0) / 9.0;
}
fn expectTNCase(fixture: *Fixture, m: usize, n: usize, k: usize, lda: usize, ldb: usize, ldc: usize, alpha: f64, beta: f64) !void {
    const allocator = std.testing.allocator;
    const a = try allocator.alloc(f64, lda * m);
    defer allocator.free(a);
    const b = try allocator.alloc(f64, ldb * n);
    defer allocator.free(b);
    const output = try allocator.alloc(f64, ldc * n);
    defer allocator.free(output);
    const expected = try allocator.alloc(f64, ldc * n);
    defer allocator.free(expected);
    @memset(a, -999);
    @memset(b, -999);
    @memset(output, -999);
    for (0..m) |i| for (0..k) |q| {
        a[q + i * lda] = sampleTN(q, i, 23);
    };
    for (0..n) |j| {
        for (0..k) |q| b[q + j * ldb] = sampleTN(q, j, 29);
        for (0..m) |i| output[i + j * ldc] = sampleTN(i, j, 31);
    }
    @memcpy(expected, output);
    tn_reference(f64, false, m, n, k, alpha, a, @intCast(lda), b, @intCast(ldb), beta, expected, @intCast(ldc));
    const before = submissions;
    try native.matrixMultiplyF64TN(&fixture.context, tnArgs(a, b, output, @intCast(m), @intCast(n), @intCast(k), @intCast(lda), @intCast(ldb), @intCast(ldc), alpha, beta), &fixture.report);
    try std.testing.expectEqual(@as(usize, if (k == 0 or alpha == 0) 1 else 2), submissions - before);
    try std.testing.expectEqual(execution.OutputState.complete, fixture.report.output);
    for (0..n) |j| for (0..ldc) |i| {
        const index = i + j * ldc;
        if (i < m) try std.testing.expectApproxEqAbs(expected[index], output[index], 1e-10) else try std.testing.expectEqual(expected[index], output[index]);
    };
    try std.testing.expectEqual(@as(usize, 0), live_scratch);
}

test "native TN reuses original Fortran oracle across padded tails large K and tiled packing" {
    const cases = [_][6]usize{
        .{ 1, 1, 1, 3, 4, 5 },
        .{ 1, 9, 19, 23, 25, 5 },
        .{ 9, 1, 19, 23, 25, 13 },
        .{ 5, 7, 3, 5, 5, 11 },
        .{ 35, 37, 67, 71, 73, 43 },
        .{ 37, 35, 71, 73, 79, 43 },
        .{ 127, 129, 65, 69, 71, 137 },
        .{ 17, 19, 2049, 2053, 2057, 23 },
        .{ 193, 65, 577, 581, 583, 199 },
        .{ 65, 33, 1025, 1029, 1031, 71 },
    };
    for ([_]u32{ 1, 2 }) |limit| {
        var fixture: Fixture = .{};
        try fixture.init(limit);
        defer fixture.deinit();
        for (cases) |shape| {
            allocations = 0;
            submissions = 0;
            try expectTNCase(&fixture, shape[0], shape[1], shape[2], shape[3], shape[4], shape[5], 0.75, -0.25);
            try std.testing.expectEqual(@as(usize, 5), allocations);
            try expectBulkDiagnostic(fixture.report, 2);
            try std.testing.expectEqual(@as(u64, @min(limit, shape[0] * shape[1])), fixture.report.provider.callbacks_completed);
        }
    }
}

test "native TN scaling empty and beta zero preserve provider semantics and NaN no-read" {
    for ([_]u32{ 1, 2 }) |limit| {
        var fixture: Fixture = .{};
        try fixture.init(limit);
        defer fixture.deinit();
        var output = [_]f64{ std.math.nan(f64), std.math.nan(f64), 99, std.math.nan(f64), std.math.nan(f64), 99 };
        try native.matrixMultiplyF64TN(&fixture.context, tnArgs(&.{}, &.{}, &output, 2, 2, 0, 0, 0, 3, 1, 0), &fixture.report);
        try std.testing.expectEqualSlices(f64, &.{ 0, 0, 99, 0, 0, 99 }, &output);
        const nan: [4]f64 = @splat(std.math.nan(f64));
        output = .{ 2, 3, 99, 4, 5, 99 };
        try native.matrixMultiplyF64TN(&fixture.context, tnArgs(&nan, &nan, &output, 2, 2, 2, 2, 2, 3, 0, 2), &fixture.report);
        try std.testing.expectEqualSlices(f64, &.{ 4, 6, 99, 8, 10, 99 }, &output);
        try std.testing.expectEqual(@as(usize, 2), submissions);
        try std.testing.expectEqual(@as(usize, 4), allocations);
        output = .{ std.math.nan(f64), std.math.nan(f64), 99, std.math.nan(f64), std.math.nan(f64), 99 };
        const finite = [_]f64{ 1, 2, 3, 4 };
        try native.matrixMultiplyF64TN(&fixture.context, tnArgs(&finite, &finite, &output, 2, 2, 2, 2, 2, 3, 1, 0), &fixture.report);
        try std.testing.expectEqualSlices(f64, &.{ 5, 11, 99, 11, 25, 99 }, &output);
        try expectTNCase(&fixture, 5, 7, 11, 13, 13, 11, 0, -0.25);
        try expectTNCase(&fixture, 5, 7, 0, 3, 3, 11, 1, 2);
        var singleton = [_]f64{std.math.nan(f64)};
        try native.matrixMultiplyF64TN(&fixture.context, tnArgs(&.{2}, &.{3}, &singleton, 1, 1, 1, 1, 1, 1, 1, 0), &fixture.report);
        try std.testing.expectEqual(@as(f64, 6), singleton[0]);
        const before = allocations;
        const before_submit = submissions;
        try native.matrixMultiplyF64TN(&fixture.context, tnArgs(&.{}, &.{}, &.{}, 0, 2, 0, 0, 0, 0, 1, 0), &fixture.report);
        try std.testing.expectEqual(execution.OutputState.unchanged, fixture.report.output);
        try std.testing.expectEqual(before, allocations);
        try std.testing.expectEqual(before_submit, submissions);
    }
}

test "native TN each provider allocation failure preserves C and terminal allocation diagnostics" {
    for ([_]u32{ 1, 2 }) |limit| for ([_]i32{ c.HETERIS_BUDGET_EXCEEDED, c.HETERIS_OUT_OF_MEMORY }) |status| {
        for (1..6) |site| {
            var fixture: Fixture = .{};
            try fixture.init(limit);
            defer fixture.deinit();
            fail_allocation = site;
            allocation_status = status;
            var output = [_]f64{ 7, 7, 7, 7 };
            const input = [_]f64{ 1, 2, 3, 4 };
            const expected: native.NativeError = if (status == c.HETERIS_BUDGET_EXCEEDED) error.ResourceBudgetExceeded else error.OutOfMemory;
            try std.testing.expectError(expected, native.matrixMultiplyF64TN(&fixture.context, tnArgs(&input, &input, &output, 2, 2, 2, 2, 2, 2, 1, 1), &fixture.report));
            try std.testing.expectEqualSlices(f64, &.{ 7, 7, 7, 7 }, &output);
            try std.testing.expectEqualDeep(execution.CallReport{ .provider = .{ .stage = .scratch_allocate, .returned_status = status } }, fixture.report);
            try std.testing.expectEqual(site, allocations);
            try std.testing.expectEqual(@as(usize, if (site <= 3) 0 else 1), submissions);
            try std.testing.expectEqual(@as(usize, 0), live_scratch);
            try std.testing.expectEqual(@as(usize, 0), fixture.context.snapshot().active_operations);
            try std.testing.expectEqual(@as(usize, 0), fixture.context.snapshot().resource_pins);
            fail_allocation = 0;
            // An explicit subsequent request proves cleanup; no implicit replay.
            try native.matrixMultiplyF64TN(&fixture.context, tnArgs(&input, &input, &output, 2, 2, 2, 2, 2, 2, 1, 1), &fixture.report);
            try std.testing.expectEqualSlices(f64, &.{ 12, 18, 18, 32 }, &output);
        }
    };
}

test "native TN both bulks preserve exact rejection execution and callback diagnostics without replay" {
    for ([_]u32{ 1, 2 }) |limit| for ([_]BulkFailure{ .reject, .execution, .callback }) |failure| {
        for (1..3) |phase| {
            var fixture: Fixture = .{};
            try fixture.init(limit);
            defer fixture.deinit();
            fail_submission = phase;
            bulk_failure = failure;
            var output = [_]f64{ 7, 7, 7, 7 };
            const input = [_]f64{ 1, 2, 3, 4 };
            const expected: native.NativeError = switch (failure) {
                .reject => error.ProviderRejected,
                .execution => error.ProviderExecutionFailed,
                .callback => error.ProviderCallbackFailed,
                .none => unreachable,
            };
            try std.testing.expectError(expected, native.matrixMultiplyF64TN(&fixture.context, tnArgs(&input, &input, &output, 2, 2, 2, 2, 2, 2, 1, 1), &fixture.report));
            try std.testing.expectEqual(phase, submissions);
            try expectBulkDiagnostic(fixture.report, phase);
            if (phase == 1 or failure == .reject) {
                try std.testing.expectEqual(execution.OutputState.unchanged, fixture.report.output);
                try std.testing.expectEqualSlices(f64, &.{ 7, 7, 7, 7 }, &output);
            } else {
                try std.testing.expectEqual(execution.OutputState.partial_or_invalid, fixture.report.output);
                const full = [_]f64{ 12, 18, 18, 32 };
                for (output, full) |value, complete| try std.testing.expect(value == 7 or value == complete);
            }
            try std.testing.expectEqual(@as(usize, 0), live_scratch);
            try std.testing.expectEqual(@as(usize, 0), fixture.context.snapshot().resource_pins);
        }
    };
}

test "native TN close after quiescent packing keeps one admission and packed lease through C" {
    var fixture: Fixture = .{};
    try fixture.init(2);
    defer fixture.deinit();
    close_after_packing = &fixture.context;
    var output = [_]f64{ 7, 7, 7, 7 };
    const input = [_]f64{ 1, 2, 3, 4 };
    try native.matrixMultiplyF64TN(&fixture.context, tnArgs(&input, &input, &output, 2, 2, 2, 2, 2, 2, 1, 1), &fixture.report);
    try std.testing.expect(close_was_busy);
    try std.testing.expect(second_phase_admitted);
    try std.testing.expectEqualSlices(f64, &.{ 12, 18, 18, 32 }, &output);
    try std.testing.expectEqual(@as(usize, 2), submissions);
    try expectBulkDiagnostic(fixture.report, 2);
    try std.testing.expectEqual(@as(usize, 0), live_scratch);
    try std.testing.expectEqual(@as(usize, 0), fixture.context.snapshot().active_operations);
    try std.testing.expectError(error.ContextClosing, native.matrixMultiplyF64TN(&fixture.context, tnArgs(&input, &input, &output, 2, 2, 2, 2, 2, 2, 1, 1), &fixture.report));
    // Closing one mathematical Context must leave the shared Runtime usable.
    var other = try execution.Context.initIntegrated(&fixture.source, .{}, &fixture.report);
    try native.matrixMultiplyF64TN(&other, tnArgs(&input, &input, &output, 2, 2, 2, 2, 2, 2, 1, 0), &fixture.report);
    _ = other.requestClose();
    try std.testing.expect(other.tryProgressClose(&fixture.report) == .closed);
}

test "native TN physical validation aliases transforms and closing precedence precede resources" {
    var fixture: Fixture = .{};
    try fixture.init(1);
    defer fixture.deinit();
    const input = [_]f64{ 1, 2, 3, 4, 5, 6 };
    var output = [_]f64{ 7, 7, 7, 7 };
    const request = tnArgs(&input, &input, &output, 2, 2, 3, 3, 3, 2, 1, 0);
    var invalid = request;
    invalid.left_matrix.values = input[0..5];
    try std.testing.expectError(error.BufferTooSmall, native.matrixMultiplyF64TN(&fixture.context, invalid, &fixture.report));
    invalid = request;
    invalid.left_matrix.leading_dimension = 2;
    try std.testing.expectError(error.InvalidLeadingDimension, native.matrixMultiplyF64TN(&fixture.context, invalid, &fixture.report));
    for ([_]@TypeOf(request.left_matrix.operation){ .normal, .adjoint }) |transform| {
        invalid = request;
        invalid.left_matrix.operation = transform;
        try std.testing.expectError(error.Unsupported, native.matrixMultiplyF64TN(&fixture.context, invalid, &fixture.report));
    }
    invalid = request;
    invalid.right_matrix.operation = .transposed;
    try std.testing.expectError(error.Unsupported, native.matrixMultiplyF64TN(&fixture.context, invalid, &fixture.report));
    invalid = request;
    invalid.right_matrix.row_count = 2;
    try std.testing.expectError(error.DimensionMismatch, native.matrixMultiplyF64TN(&fixture.context, invalid, &fixture.report));
    invalid = tnArgs(&output, &input, &output, 2, 2, 2, 2, 2, 2, 0, 0);
    try std.testing.expectError(error.AliasingNotAllowed, native.matrixMultiplyF64TN(&fixture.context, invalid, &fixture.report));
    invalid = tnArgs(&input, &output, &output, 2, 2, 2, 2, 2, 2, 0, 0);
    try std.testing.expectError(error.AliasingNotAllowed, native.matrixMultiplyF64TN(&fixture.context, invalid, &fixture.report));
    try std.testing.expectEqual(@as(usize, 0), allocations);
    try std.testing.expectEqual(@as(usize, 0), submissions);
    _ = fixture.context.requestClose();
    try std.testing.expectError(error.AliasingNotAllowed, native.matrixMultiplyF64TN(&fixture.context, invalid, &fixture.report));
    try std.testing.expectError(error.ContextClosing, native.matrixMultiplyF64TN(&fixture.context, request, &fixture.report));
    try std.testing.expectError(error.ContextClosing, native.matrixMultiplyF64TN(&fixture.context, tnArgs(&.{}, &.{}, &.{}, 0, 0, 0, 0, 0, 0, 1, 0), &fixture.report));
    try std.testing.expectEqualSlices(f64, &.{ 7, 7, 7, 7 }, &output);
    try std.testing.expectEqual(@as(usize, 0), allocations);
}

const grid = @import("zynum-core").native_grid;
const periodic_reference = @import("periodic-test-reference");
fn gridArgs(shape: [3]usize, input: []const f64, output: []f64) grid.PeriodicLaplacianF64 {
    return .{ .shape = shape, .input = input, .output = output, .inv_h2 = .{ 2, 3, 4 } };
}
fn expectGridCase(fixture: *Fixture, shape: [3]usize, permutation: [3]usize) !void {
    const n = shape[0] * shape[1] * shape[2];
    const input = try std.testing.allocator.alloc(f64, n + 3);
    defer std.testing.allocator.free(input);
    const output = try std.testing.allocator.alloc(f64, n + 5);
    defer std.testing.allocator.free(output);
    @memset(input, -991);
    @memset(output, 773);
    const h: [3]f64 = .{ 2, 3, 4 };
    for (0..shape[0]) |i| for (0..shape[1]) |j| for (0..shape[2]) |k| {
        const index: [3]usize = .{ i, j, k };
        input[(i * shape[1] + j) * shape[2] + k] = periodic_reference.fourierInput(index[permutation[0]], index[permutation[2]], shape[permutation[0]], shape[permutation[2]]);
    };
    const saved = try std.testing.allocator.dupe(f64, input);
    defer std.testing.allocator.free(saved);
    const before_allocations = allocations;
    const before_submissions = submissions;
    try grid.periodicLaplacianF64(&fixture.context, gridArgs(shape, input, output), &fixture.report);
    try std.testing.expectEqual(@as(usize, 2), allocations - before_allocations);
    try std.testing.expectEqual(@as(usize, 1), submissions - before_submissions);
    try std.testing.expectEqual(execution.OutputState.complete, fixture.report.output);
    try std.testing.expectEqual(execution.SubmissionState.quiescent, fixture.report.provider.state);
    for (0..shape[0]) |i| for (0..shape[1]) |j| for (0..shape[2]) |k| {
        const index: [3]usize = .{ i, j, k };
        const expected = periodic_reference.fourierExpected(index[permutation[0]], index[permutation[2]], shape[permutation[0]], shape[permutation[2]], .{ h[permutation[0]], h[permutation[1]], h[permutation[2]] });
        try std.testing.expectApproxEqAbs(expected, output[(i * shape[1] + j) * shape[2] + k], 2e-13);
    };
    try std.testing.expectEqualSlices(f64, saved, input);
    for (output[n..]) |value| try std.testing.expectEqual(@as(f64, 773), value);
    try std.testing.expectEqual(@as(usize, 0), live_scratch);
}

test "native periodic Laplacian reuses original Fourier oracle across geometry and task limits" {
    const shapes = [_][3]usize{ .{ 1, 1, 1 }, .{ 1, 2, 3 }, .{ 2, 1, 3 }, .{ 2, 3, 1 }, .{ 2, 2, 2 }, .{ 4, 3, 5 }, .{ 7, 8, 9 } };
    for ([_]u32{ 1, 2 }) |limit| {
        var fixture: Fixture = .{};
        try fixture.init(limit);
        defer fixture.deinit();
        for (shapes) |shape| for ([_][3]usize{ .{ 0, 1, 2 }, .{ 1, 2, 0 }, .{ 2, 0, 1 } }) |permutation| try expectGridCase(&fixture, shape, permutation);
        const x = [_]f64{ 1, 2, -1, 4, 3, -2, 7, 0 };
        const y = [_]f64{ 2, -1, 3, 1, -2, 4, 0, 5 };
        var lx: [8]f64 = undefined;
        var ly: [8]f64 = undefined;
        try grid.periodicLaplacianF64(&fixture.context, gridArgs(.{ 2, 2, 2 }, &x, &lx), &fixture.report);
        try grid.periodicLaplacianF64(&fixture.context, gridArgs(.{ 2, 2, 2 }, &y, &ly), &fixture.report);
        var xly: f64 = 0;
        var lxy: f64 = 0;
        for (x, lx, y, ly) |a, la, b, lb| {
            xly += a * lb;
            lxy += la * b;
        }
        try std.testing.expectEqual(xly, lxy);
        const constant: [8]f64 = @splat(1);
        try grid.periodicLaplacianF64(&fixture.context, gridArgs(.{ 2, 2, 2 }, &constant, &lx), &fixture.report);
        for (lx) |value| try std.testing.expectEqual(@as(f64, 0), value);
    }
}

test "native periodic Laplacian zero axes and extreme geometry preserve validation and admission" {
    for ([_]u32{ 1, 2 }) |limit| {
        var fixture: Fixture = .{};
        try fixture.init(limit);
        defer fixture.deinit();
        for (0..8) |bits| {
            const shape: [3]usize = .{ if ((bits & 1) == 0) 0 else 1, if ((bits & 2) == 0) 0 else 1, if ((bits & 4) == 0) 0 else 1 };
            const x = [_]f64{3};
            var y = [_]f64{9};
            allocations = 0;
            submissions = 0;
            try grid.periodicLaplacianF64(&fixture.context, gridArgs(shape, &x, &y), &fixture.report);
            try std.testing.expectEqual(@as(usize, if (bits == 7) 2 else 0), allocations);
            try std.testing.expectEqual(@as(usize, if (bits == 7) 1 else 0), submissions);
            try std.testing.expectEqual(if (bits == 7) execution.OutputState.complete else execution.OutputState.unchanged, fixture.report.output);
            try std.testing.expectEqual(@as(f64, if (bits == 7) 0 else 9), y[0]);
        }
        allocations = 0;
        submissions = 0;
        try grid.periodicLaplacianF64(&fixture.context, gridArgs(.{ std.math.maxInt(usize), 0, std.math.maxInt(usize) }, &.{}, &.{}), &fixture.report);
        try std.testing.expectError(error.ExtentOverflow, grid.periodicLaplacianF64(&fixture.context, gridArgs(.{ std.math.maxInt(usize), 2, 1 }, &.{}, &.{}), &fixture.report));
        try std.testing.expectError(error.ExtentOverflow, grid.periodicLaplacianF64(&fixture.context, gridArgs(.{ std.math.maxInt(usize), 1, 1 }, &.{}, &.{}), &fixture.report));
        _ = fixture.context.requestClose();
        try std.testing.expectError(error.ContextClosing, grid.periodicLaplacianF64(&fixture.context, gridArgs(.{ 0, 1, 1 }, &.{}, &.{}), &fixture.report));
        try std.testing.expectEqual(@as(usize, 0), allocations);
        try std.testing.expectEqual(@as(usize, 0), submissions);
    }
}

test "native periodic Laplacian bounded controls views aliases and coefficients precede resources" {
    var fixture: Fixture = .{};
    try fixture.init(1);
    defer fixture.deinit();
    const x = [_]f64{ 1, 2, 3, 4 };
    var y = [_]f64{ 7, 7, 7, 7 };
    try std.testing.expectError(error.InvalidView, grid.periodicLaplacianF64(&fixture.context, gridArgs(.{ 1, 2, 2 }, x[0..3], &y), &fixture.report));
    try std.testing.expectError(error.InvalidView, grid.periodicLaplacianF64(&fixture.context, gridArgs(.{ 1, 2, 2 }, &x, y[0..3]), &fixture.report));
    try std.testing.expectError(error.AliasingNotAllowed, grid.periodicLaplacianF64(&fixture.context, gridArgs(.{ 0, 2, 2 }, y[0..3], y[2..]), &fixture.report));
    for ([_]f64{ 0, -0.0, -1, std.math.nan(f64), std.math.inf(f64) }) |h| for (0..3) |axis| {
        var request = gridArgs(.{ 0, 2, 2 }, &x, &y);
        request.inv_h2[axis] = h;
        try std.testing.expectError(error.InvalidCoefficient, grid.periodicLaplacianF64(&fixture.context, request, &fixture.report));
    };
    _ = fixture.context.requestClose();
    var request = gridArgs(.{ 0, 2, 2 }, &x, &y);
    request.inv_h2[0] = -1;
    try std.testing.expectError(error.InvalidCoefficient, grid.periodicLaplacianF64(&fixture.context, request, &fixture.report));
    try std.testing.expectEqualDeep(execution.CallReport{}, fixture.report);
    try std.testing.expectEqualSlices(f64, &.{ 7, 7, 7, 7 }, &y);
    try std.testing.expectEqual(@as(usize, 0), allocations);
    try std.testing.expectEqual(@as(usize, 0), submissions);
    // Opaque C controls, null capacities and uncheckable declared ranges are
    // exercised by the independent C consumer, not by invalid Zig slices.
}

test "native periodic Laplacian allocation rejection and quiescent failures preserve raw diagnostics" {
    for ([_]u32{ 1, 2 }) |limit| {
        for ([_]i32{ c.HETERIS_BUDGET_EXCEEDED, c.HETERIS_OUT_OF_MEMORY }) |status| for (1..3) |site| {
            var fixture: Fixture = .{};
            try fixture.init(limit);
            defer fixture.deinit();
            const x = [_]f64{ 1, 2, 3, 4 };
            var y = [_]f64{ 7, 7, 7, 7 };
            fail_allocation = site;
            allocation_status = status;
            const expected: grid.Error = if (status == c.HETERIS_BUDGET_EXCEEDED) error.ResourceBudgetExceeded else error.OutOfMemory;
            try std.testing.expectError(expected, grid.periodicLaplacianF64(&fixture.context, gridArgs(.{ 1, 2, 2 }, &x, &y), &fixture.report));
            try std.testing.expectEqualSlices(f64, &.{ 7, 7, 7, 7 }, &y);
            try std.testing.expectEqualDeep(execution.CallReport{ .provider = .{ .stage = .scratch_allocate, .returned_status = status } }, fixture.report);
            try std.testing.expectEqual(site, allocations);
            try std.testing.expectEqual(@as(usize, 0), submissions);
            try std.testing.expectEqual(@as(usize, 0), live_scratch);
            fail_allocation = 0;
            try grid.periodicLaplacianF64(&fixture.context, gridArgs(.{ 1, 2, 2 }, &x, &y), &fixture.report);
            try std.testing.expectEqual(@as(usize, 1), submissions);
        };
        for ([_]BulkFailure{ .reject, .execution, .callback }) |failure| {
            var fixture: Fixture = .{};
            try fixture.init(limit);
            defer fixture.deinit();
            const x = [_]f64{ 1, 2, 3, 4 };
            var y = [_]f64{ 7, 7, 7, 7 };
            fail_submission = 1;
            bulk_failure = failure;
            const expected: grid.Error = switch (failure) {
                .reject => error.ProviderRejected,
                .execution => error.ProviderExecutionFailed,
                .callback => error.ProviderCallbackFailed,
                .none => unreachable,
            };
            try std.testing.expectError(expected, grid.periodicLaplacianF64(&fixture.context, gridArgs(.{ 1, 2, 2 }, &x, &y), &fixture.report));
            try expectBulkDiagnostic(fixture.report, 1);
            try std.testing.expectEqual(@as(usize, 1), submissions);
            try std.testing.expectEqual(@as(usize, 0), live_scratch);
            try std.testing.expectEqual(if (failure == .reject) execution.OutputState.unchanged else execution.OutputState.partial_or_invalid, fixture.report.output);
            if (failure == .reject) try std.testing.expectEqualSlices(f64, &.{ 7, 7, 7, 7 }, &y);
        }
    }
}
const GridWorker = struct {
    context: *execution.Context,
    nonempty_error: ?grid.Error = null,
    empty_ok: bool = false,
    fn run(raw: ?*anyopaque, begin: u64, end: u64) callconv(.c) i32 {
        if (begin != 0 or end != 1) return c.HETERIS_INVALID_ARGUMENT;
        const self: *@This() = @ptrCast(@alignCast(raw.?));
        const x = [_]f64{1};
        var y = [_]f64{7};
        var report: execution.CallReport = .{};
        grid.periodicLaplacianF64(self.context, gridArgs(.{ 1, 1, 1 }, &x, &y), &report) catch |err| {
            self.nonempty_error = err;
        };
        grid.periodicLaplacianF64(self.context, gridArgs(.{ 0, 1, 1 }, &.{}, &.{}), &report) catch return c.HETERIS_INVALID_ARGUMENT;
        self.empty_ok = report.output == .unchanged and report.provider.state == .not_submitted and y[0] == 7;
        return c.HETERIS_OK;
    }
};
test "native periodic Laplacian admitted close and managed worker paths preserve lifecycle" {
    {
        var fixture: Fixture = .{};
        try fixture.init(2);
        defer fixture.deinit();
        grid_close_during = &fixture.context;
        const x = [_]f64{ 1, 2, 3, 4 };
        var y = [_]f64{ 7, 7, 7, 7 };
        try grid.periodicLaplacianF64(&fixture.context, gridArgs(.{ 1, 2, 2 }, &x, &y), &fixture.report);
        try std.testing.expect(grid_close_during_busy.load(.acquire));
        try std.testing.expectEqual(@as(usize, 1), submissions);
        try std.testing.expectEqual(@as(usize, 0), live_scratch);
        try std.testing.expectEqual(@as(usize, 0), fixture.context.snapshot().active_operations);
        try std.testing.expectError(error.ContextClosing, grid.periodicLaplacianF64(&fixture.context, gridArgs(.{ 1, 2, 2 }, &x, &y), &fixture.report));
        var other = try execution.Context.initIntegrated(&fixture.source, .{}, &fixture.report);
        grid_close_during = null;
        try grid.periodicLaplacianF64(&other, gridArgs(.{ 1, 2, 2 }, &x, &y), &fixture.report);
        _ = other.requestClose();
        try std.testing.expect(other.tryProgressClose(&fixture.report) == .closed);
    }
    {
        var fixture: Fixture = .{};
        try fixture.init(1);
        defer fixture.deinit();
        var worker: GridWorker = .{ .context = &fixture.context };
        var outcome: c.heteris_rp_bulk_result_v1 = undefined;
        try std.testing.expectEqual(@as(i32, 0), original.cpu_bulk_sync.?(fixture.source.provider, 0, 1, 1, 1, GridWorker.run, &worker, &outcome));
        try std.testing.expectEqual(@as(?grid.Error, error.WouldDeadlock), worker.nonempty_error);
        try std.testing.expect(worker.empty_ok);
        try std.testing.expectEqual(@as(u32, c.HETERIS_RP_QUIESCENT), outcome.state);
        try std.testing.expectEqual(@as(usize, 0), live_scratch);
    }
}

test "native periodic Laplacian IEEE outputs and capacity tails retain direct output semantics" {
    var fixture: Fixture = .{};
    try fixture.init(1);
    defer fixture.deinit();
    for ([_]f64{ std.math.nan(f64), std.math.inf(f64), std.math.floatMax(f64) }) |value| {
        const x = [_]f64{ value, 991 };
        var y = [_]f64{ 7, 773 };
        try grid.periodicLaplacianF64(&fixture.context, gridArgs(.{ 1, 1, 1 }, &x, &y), &fixture.report);
        try std.testing.expect(!std.math.isFinite(y[0]));
        try std.testing.expectEqual(@as(f64, 773), y[1]);
        try std.testing.expectEqual(execution.OutputState.complete, fixture.report.output);
    }
    const x = [_]f64{ 1, 0 };
    var y = [_]f64{ 7, 7, 773 };
    var request = gridArgs(.{ 1, 1, 2 }, &x, &y);
    request.inv_h2[2] = std.math.floatMax(f64);
    try grid.periodicLaplacianF64(&fixture.context, request, &fixture.report);
    try std.testing.expect(std.math.isInf(y[0]) and std.math.isInf(y[1]));
    try std.testing.expectEqual(@as(f64, 773), y[2]);
    try std.testing.expectEqual(execution.OutputState.complete, fixture.report.output);
}

// Additive spectral source coverage: existing fixture/provider/Context are reused.
// These tests enter the trusted Core seam with actual typed objects/ranges;
// control preflight and passive guard ABI behavior are exercised by C/C++ cases.
const spectral = @import("zynum-core").native_spectral;
const SpectralSpan = struct {
    lease: c.heteris_rp_scratch_v1,
    values: []f64,
    fn init(fixture: *Fixture, count: usize) !SpectralSpan {
        var lease: c.heteris_rp_scratch_v1 = undefined;
        try std.testing.expectEqual(@as(i32, 0), fixture.api.scratch_allocate.?(fixture.source.provider, count * @sizeOf(f64), @alignOf(f64), &lease));
        const pointer: [*]f64 = @ptrCast(@alignCast(lease.data.?));
        const values = pointer[0..count];
        @memset(values, 991);
        return .{ .lease = lease, .values = values };
    }
    fn deinit(self: *SpectralSpan, fixture: *Fixture) void {
        if (fixture.api.scratch_release.?(self.lease.lease) != 0) @panic("spectral fixture scratch release");
    }
};
fn spectralBuffer(values: []const f64, ld: u64) spectral.ConstBuffer {
    return .{ .address = @intFromPtr(values.ptr), .backing = @intCast(values.len), .ld = ld };
}
const SpectralProbe = struct {
    diagonal: [3]f64 = .{ -2, 1, 4 },
    dense: ?[9]f64 = null,
    calls: u64 = 0,
    failure_order: u64 = 0,
    failure_code: i32 = 0,
    nonfinite_order: u64 = 0,
    close_order: u64 = 0,
    busy_during_close: bool = false,
    nested: bool = false,
    nested_admitted: bool = false,
    owner: std.Thread.Id,
    fn apply(raw: ?*anyopaque, context: *execution.Context, input: spectral.ConstPanel, output: spectral.Panel) i32 {
        const self: *SpectralProbe = @ptrCast(@alignCast(raw.?));
        if (std.Thread.getCurrentId() != self.owner) @panic("spectral loop left its synchronous caller");
        self.calls += 1;
        if (input.rows != output.rows or input.columns != output.columns or input.ld != input.rows or output.ld != input.rows) @panic("spectral callback geometry");
        if (self.nested) {
            var child = context.begin() catch return -28001;
            self.nested_admitted = context.snapshot().active_operations == 2;
            child.end();
        }
        if (self.calls == self.close_order) {
            _ = context.requestClose();
            var report: execution.CallReport = .{};
            self.busy_during_close = context.tryProgressClose(&report) == .busy;
        }
        if (self.calls == self.failure_order) return self.failure_code;
        if (input.rows == 0 or input.columns == 0) {
            if (input.data != null or output.data != null) @panic("empty callback formed pointers");
            return 0;
        }
        for (0..@intCast(input.columns)) |j| for (0..@intCast(input.rows)) |i| {
            if (self.dense) |matrix| {
                var product: f128 = 0;
                for (0..3) |k| product += @as(f128, matrix[k * 3 + i]) * input.cell(k, j).*;
                output.cell(i, j).* = @floatCast(product);
            } else output.cell(i, j).* = self.diagonal[i % self.diagonal.len] * input.cell(i, j).*;
        };
        if (self.calls == self.nonfinite_order) output.cell(0, 0).* = std.math.nan(f64);
        return 0;
    }
};
fn spectralRequest(kind: spectral.Kind, probe: *SpectralProbe, rows: u64, columns: u64, orders: u64, x: []const f64, ldx: u64, coefficients: []const f64, y: []f64, ldy: u64, work: []f64, lower: f64, upper: f64) spectral.Args {
    return .{ .kind = kind, .rows = rows, .columns = columns, .orders = orders, .lower = lower, .upper = upper, .input = spectralBuffer(x, ldx), .coefficients = spectralBuffer(coefficients, orders), .output = spectralBuffer(y, ldy), .work = spectralBuffer(work, rows), .operator = .{ .apply = SpectralProbe.apply, .data = probe } };
}
// Closed-form degree <=3 polynomials, independent of the production recurrence.
fn spectralPolynomial(order: usize, value: f128) f128 {
    return switch (order) {
        0 => 1,
        1 => value,
        2 => 2 * value * value - 1,
        3 => 4 * value * value * value - 3 * value,
        else => unreachable,
    };
}

test "native spectral diagonal polynomial and raw moments match independent closed forms with padded borrowed panels" {
    var fixture: Fixture = .{};
    try fixture.init(2);
    defer fixture.deinit();
    var scratch = try SpectralSpan.init(&fixture, 32);
    defer scratch.deinit(&fixture);
    var probe: SpectralProbe = .{ .owner = std.Thread.getCurrentId(), .nested = true };
    const x = [_]f64{ 1, 2, 3, std.math.nan(f64), 991, -2, 0.5, 4, std.math.nan(f64), 991 };
    const coefficients = [_]f64{ 0.5, -0.75, 0.25, 0.125 };
    var output: [10]f64 = @splat(773);
    var report: spectral.Report = .{};
    var request = spectralRequest(.series, &probe, 3, 2, 4, &x, 5, &coefficients, &output, 4, scratch.values, -3, 5);
    try std.testing.expectEqual(spectral.Status.ok, spectral.invoke(&fixture.context, request, &report));
    for (0..2) |j| for (0..3) |i| {
        const scaled = (@as(f128, probe.diagonal[i]) - 1) / 4;
        var expected: f128 = 0;
        for (coefficients, 0..) |coefficient, k| expected += @as(f128, coefficient) * spectralPolynomial(k, scaled) * x[j * 5 + i];
        try std.testing.expectApproxEqAbs(@as(f64, @floatCast(expected)), output[j * 4 + i], 1e-13);
    };
    try std.testing.expectEqual(@as(f64, 773), output[3]);
    try std.testing.expectEqual(@as(f64, 773), output[7]);
    try std.testing.expectEqual(@as(f64, 773), output[8]);
    try std.testing.expectEqual(@as(u64, 3), probe.calls);
    try std.testing.expect(probe.nested_admitted and report.admission_succeeded and report.progress.callback_returned);
    try std.testing.expectEqual(@as(usize, 0), fixture.context.snapshot().active_operations);
    try std.testing.expectEqual(@as(usize, 0), submissions);
    probe.calls = 0;
    @memset(&output, 773);
    request.kind = .moments;
    request.orders = 4;
    request.output = spectralBuffer(&output, 5);
    try std.testing.expectEqual(spectral.Status.ok, spectral.invoke(&fixture.context, request, &report));
    for (0..2) |j| for (0..4) |k| {
        var expected: f128 = 0;
        for (0..3) |i| {
            const scaled = (@as(f128, probe.diagonal[i]) - 1) / 4;
            expected += @as(f128, x[j * 5 + i]) * x[j * 5 + i] * spectralPolynomial(k, scaled);
        }
        try std.testing.expectApproxEqAbs(@as(f64, @floatCast(expected)), output[j * 5 + k], 1e-12);
    };
    try std.testing.expectEqual(@as(f64, 773), output[4]);
    try std.testing.expectEqual(@as(f64, 773), output[9]);
    for (scratch.values[26..]) |value| try std.testing.expectEqual(@as(f64, 991), value);
    try std.testing.expectEqual(@as(u64, 3), probe.calls);
    try std.testing.expectEqual(@as(u64, 3), report.progress.callback_order);
}

test "native spectral extreme and adjacent intervals preserve widened affine and accumulation arithmetic" {
    var fixture: Fixture = .{};
    try fixture.init(1);
    defer fixture.deinit();
    var scratch = try SpectralSpan.init(&fixture, 8);
    defer scratch.deinit(&fixture);
    var probe: SpectralProbe = .{ .owner = std.Thread.getCurrentId() };
    var output = [_]f64{ 773, 773 };
    const input = [_]f64{1};
    const coefficients = [_]f64{ 0, 1 };
    var report: spectral.Report = .{};
    // Exact adjacent f64 endpoints. A premature f64 midpoint would be 1.
    const adjacent: f64 = @bitCast(@as(u64, @bitCast(@as(f64, 1))) + 1);
    probe.diagonal = .{ 1, 1, 1 };
    var request = spectralRequest(.series, &probe, 1, 1, 2, &input, 1, &coefficients, &output, 1, scratch.values, 1, adjacent);
    try std.testing.expectEqual(spectral.Status.ok, spectral.invoke(&fixture.context, request, &report));
    try std.testing.expectEqual(@as(f64, -1), output[0]);
    probe.diagonal[0] = 0;
    request.lower = -std.math.floatMax(f64);
    request.upper = std.math.floatMax(f64);
    try std.testing.expectEqual(spectral.Status.ok, spectral.invoke(&fixture.context, request, &report));
    try std.testing.expectEqual(@as(f64, 0), output[0]);
    // Endpoint action and representable cancellation with extreme coefficients.
    probe.diagonal[0] = std.math.floatMax(f64);
    const cancelling = [_]f64{ std.math.floatMax(f64), -std.math.floatMax(f64) };
    request.coefficients = spectralBuffer(&cancelling, 2);
    request.lower = std.math.floatMax(f64) / 2;
    request.upper = std.math.floatMax(f64);
    try std.testing.expectEqual(spectral.Status.ok, spectral.invoke(&fixture.context, request, &report));
    try std.testing.expectEqual(@as(f64, 0), output[0]);
    try std.testing.expectEqual(@as(f64, 773), output[1]);
}

test "native spectral moment row order retains wide cancellation and narrows only the final dot" {
    var fixture: Fixture = .{};
    try fixture.init(1);
    defer fixture.deinit();
    var scratch = try SpectralSpan.init(&fixture, 13);
    defer scratch.deinit(&fixture);
    var probe: SpectralProbe = .{ .owner = std.Thread.getCurrentId(), .diagonal = .{ 0x1p53, 1, -0x1p53 } };
    const input = [_]f64{ 1, 1, 1 };
    var output = [_]f64{ 773, 773, 773 };
    var report: spectral.Report = .{};
    const request = spectralRequest(.moments, &probe, 3, 1, 2, &input, 3, &.{}, &output, 2, scratch.values, -0x1p53, 0x1p53);
    try std.testing.expectEqual(spectral.Status.ok, spectral.invoke(&fixture.context, request, &report));
    try std.testing.expectEqual(@as(f64, 3), output[0]);
    try std.testing.expectEqual(@as(f64, 0x1p-53), output[1]);
    try std.testing.expectEqual(@as(f64, 773), output[2]);
}

test "native spectral numerical and raw callback failures preserve public bytes and distinguish returned status" {
    var fixture: Fixture = .{};
    try fixture.init(1);
    defer fixture.deinit();
    var scratch = try SpectralSpan.init(&fixture, 16);
    defer scratch.deinit(&fixture);
    var probe: SpectralProbe = .{ .owner = std.Thread.getCurrentId(), .diagonal = .{ 1, 1, 1 } };
    var input = [_]f64{ 2, 3 };
    var coefficients = [_]f64{ 1, 2, 3 };
    var output = [_]f64{ 773, 773, 773, 773 };
    const saved = output;
    var report: spectral.Report = .{};
    var request = spectralRequest(.series, &probe, 2, 1, 3, &input, 2, &coefficients, &output, 2, scratch.values, -1, 1);
    for ([_]i32{ std.math.minInt(i32), -1907, 1, 8, 10, std.math.maxInt(i32) }) |code| {
        probe.calls = 0;
        probe.failure_order = 2;
        probe.failure_code = code;
        try std.testing.expectEqual(spectral.Status.callback_failure, spectral.invoke(&fixture.context, request, &report));
        try std.testing.expectEqualSlices(f64, &saved, &output);
        try std.testing.expectEqual(spectral.Phase.apply, report.phase);
        try std.testing.expectEqual(code, report.progress.callback_status);
        try std.testing.expectEqual(@as(u64, 2), report.progress.callbacks_returned);
    }
    probe.failure_order = 0;
    probe.calls = 0;
    probe.nonfinite_order = 1;
    try std.testing.expectEqual(spectral.Status.numerical_failure, spectral.invoke(&fixture.context, request, &report));
    try std.testing.expectEqual(spectral.Phase.recurrence, report.phase);
    try std.testing.expect(report.progress.callback_returned and report.progress.callback_status == 0);
    probe.nonfinite_order = 0;
    request.orders = 1;
    coefficients[0] = std.math.floatMax(f64);
    try std.testing.expectEqual(spectral.Status.numerical_failure, spectral.invoke(&fixture.context, request, &report));
    try std.testing.expectEqual(spectral.Phase.initialize, report.phase);
    try std.testing.expect(!report.progress.callback_returned and report.progress.order == 0);
    coefficients[0] = 1;
    input[0] = std.math.floatMax(f64);
    request.kind = .moments;
    try std.testing.expectEqual(spectral.Status.numerical_failure, spectral.invoke(&fixture.context, request, &report));
    try std.testing.expectEqual(spectral.Phase.moment, report.phase);
    try std.testing.expect(!report.progress.callback_returned);
    try std.testing.expectEqualSlices(f64, &saved, &output);
}

test "native spectral admission counter and closing failures preserve validation and callback validity" {
    var fixture: Fixture = .{};
    try fixture.init(1);
    defer fixture.deinit();
    var scratch = try SpectralSpan.init(&fixture, 8);
    defer scratch.deinit(&fixture);
    var probe: SpectralProbe = .{ .owner = std.Thread.getCurrentId(), .diagonal = .{ 1, 1, 1 } };
    var output = [_]f64{773};
    var report: spectral.Report = .{};
    const request = spectralRequest(.series, &probe, 1, 1, 3, &.{1}, 1, &.{ 1, 2, 3 }, &output, 1, scratch.values, -1, 1);
    execution.Context.testing.configure(&fixture.context, 0, null);
    try std.testing.expectEqual(spectral.Status.context_failure, spectral.invoke(&fixture.context, request, &report));
    try std.testing.expectEqual(spectral.Admission.counter_overflow, report.admission);
    try std.testing.expect(report.admission_attempted and !report.admission_succeeded and !report.progress.order_valid and !report.progress.callback_returned);
    execution.Context.testing.configure(&fixture.context, std.math.maxInt(usize), null);
    probe.close_order = 2;
    try std.testing.expectEqual(spectral.Status.ok, spectral.invoke(&fixture.context, request, &report));
    try std.testing.expect(probe.busy_during_close and report.output_complete and report.admission_succeeded);
    try std.testing.expectEqual(@as(f64, 6), output[0]);
    try std.testing.expectEqual(spectral.Status.context_failure, spectral.invoke(&fixture.context, request, &report));
    try std.testing.expectEqual(spectral.Admission.closing, report.admission);
    try std.testing.expect(!report.progress.callback_returned and !report.progress.order_valid);
    try std.testing.expectEqual(@as(usize, 0), fixture.context.snapshot().active_operations);
}

test "native spectral dense symmetric degree two agrees with direct matrix polynomial and moments" {
    var fixture: Fixture = .{};
    try fixture.init(1);
    defer fixture.deinit();
    var scratch = try SpectralSpan.init(&fixture, 13);
    defer scratch.deinit(&fixture);
    const matrix = [9]f64{ 1, 0.25, 0, 0.25, 2, -0.5, 0, -0.5, 3 };
    var probe: SpectralProbe = .{ .owner = std.Thread.getCurrentId(), .dense = matrix };
    const x = [_]f64{ 1, -2, 0.5 };
    const coefficients = [_]f64{ 0.5, -0.75, 0.125 };
    var y = [_]f64{ 773, 773, 773, 773 };
    // Gershgorin intervals lie in [0,4]. Build K and K^2 directly; no
    // production recurrence or recurrence_arithmetic helper enters the oracle.
    var k_matrix: [9]f128 = undefined;
    var t2_matrix: [9]f128 = undefined;
    for (0..3) |j| for (0..3) |i| {
        k_matrix[j * 3 + i] = (@as(f128, matrix[j * 3 + i]) - (if (i == j) @as(f128, 2) else 0)) / 2;
    };
    for (0..3) |j| for (0..3) |i| {
        var squared: f128 = 0;
        for (0..3) |k| squared += k_matrix[k * 3 + i] * k_matrix[j * 3 + k];
        t2_matrix[j * 3 + i] = 2 * squared - (if (i == j) @as(f128, 1) else 0);
    };
    var reference: [3][3]f128 = undefined;
    for (0..3) |i| {
        reference[0][i] = x[i];
        reference[1][i] = 0;
        reference[2][i] = 0;
        for (0..3) |j| {
            reference[1][i] += k_matrix[j * 3 + i] * x[j];
            reference[2][i] += t2_matrix[j * 3 + i] * x[j];
        }
    }
    var report: spectral.Report = .{};
    var request = spectralRequest(.series, &probe, 3, 1, 3, &x, 3, &coefficients, &y, 3, scratch.values, 0, 4);
    try std.testing.expectEqual(spectral.Status.ok, spectral.invoke(&fixture.context, request, &report));
    for (0..3) |i| {
        var expected: f128 = 0;
        for (coefficients, 0..) |coefficient, k| expected += @as(f128, coefficient) * reference[k][i];
        try std.testing.expectApproxEqAbs(@as(f64, @floatCast(expected)), y[i], 1e-14);
    }
    request.kind = .moments;
    try std.testing.expectEqual(spectral.Status.ok, spectral.invoke(&fixture.context, request, &report));
    for (0..3) |k| {
        var expected: f128 = 0;
        for (0..3) |i| expected += @as(f128, x[i]) * reference[k][i];
        try std.testing.expectApproxEqAbs(@as(f64, @floatCast(expected)), y[k], 1e-14);
    }
    try std.testing.expectEqual(@as(f64, 773), y[3]);
    try std.testing.expectEqual(@as(f64, 991), scratch.values[12]);
}

// Two independent produced-overflow sites, appended under the narrow repair
// window. Duplicate input columns give real output padding as well as tails.
test "native spectral finite shifted state reports final series accumulation overflow without publishing bytes" {
    var fixture: Fixture = .{};
    try fixture.init(1);
    defer fixture.deinit();
    var scratch = try SpectralSpan.init(&fixture, 10);
    defer scratch.deinit(&fixture);
    var probe: SpectralProbe = .{ .owner = std.Thread.getCurrentId(), .diagonal = .{ 1, 1, 1 } };
    const input = [_]f64{ 1e308, 1e308 };
    const coefficients = [_]f64{ 1, 1 };
    const sentinel_nan: f64 = @bitCast(@as(u64, 0x7ff8000000000371));
    var output = [_]f64{ 773, sentinel_nan, -0.0, 773, sentinel_nan, -0.0, 991, 991 };
    const saved = output;
    var report: spectral.Report = .{};
    // H=I, [-1,1]: initial sum=x and T1=x are finite, but x+x>MAX.
    // No production arithmetic helper is used to derive this expectation.
    try std.testing.expect(std.math.isFinite(input[0]));
    try std.testing.expect(@as(f128, 2) * input[0] > @as(f128, std.math.floatMax(f64)));
    const request = spectralRequest(.series, &probe, 1, 2, 2, &input, 1, &coefficients, &output, 3, scratch.values, -1, 1);
    try std.testing.expectEqual(spectral.Status.numerical_failure, spectral.invoke(&fixture.context, request, &report));
    // Workspace layout is the documented previous/current/next/sum prefix.
    // The first shifted value was produced and stored before accumulation failed.
    try std.testing.expectEqual(input[0], scratch.values[4]);
    try std.testing.expect(std.math.isFinite(scratch.values[4]));
    try std.testing.expectEqual(input[0], scratch.values[6]);
    try std.testing.expectEqual(spectral.Phase.recurrence, report.phase);
    try std.testing.expect(report.admission_attempted and report.admission_succeeded and !report.output_complete);
    try std.testing.expect(report.progress.order_valid and report.progress.order == 1);
    try std.testing.expect(report.progress.callback_returned and report.progress.callback_order == 1);
    try std.testing.expectEqual(@as(i32, 0), report.progress.callback_status);
    try std.testing.expectEqual(@as(u64, 1), report.progress.callbacks_returned);
    try std.testing.expectEqual(@as(u64, 1), probe.calls);
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&saved), std.mem.asBytes(&output));
    try std.testing.expectEqual(@as(usize, 0), fixture.context.snapshot().active_operations);
    try std.testing.expectEqual(@as(usize, 0), submissions);
}

const SpectralDyadicShiftProbe = struct {
    calls: u64 = 0,
    raw_first: f64 = 0,
    raw_second: f64 = 0,
    raw_finite: bool = true,
    returned_status: i32 = -1,
    owner: std.Thread.Id,
    fn apply(raw: ?*anyopaque, context: *execution.Context, input: spectral.ConstPanel, output: spectral.Panel) i32 {
        const self: *SpectralDyadicShiftProbe = @ptrCast(@alignCast(raw.?));
        if (std.Thread.getCurrentId() != self.owner or context.snapshot().active_operations != 1) @panic("dyadic spectral callback caller/admission");
        if (input.rows != 2 or output.rows != 2 or input.columns != output.columns or input.ld != 2 or output.ld != 2) @panic("dyadic spectral callback geometry");
        self.calls += 1;
        for (0..@intCast(input.columns)) |j| {
            // Compute the two finite products separately. Factoring this as
            // (11/32)*(x0+x1) would overflow inside H and test the wrong site.
            const left: f64 = (11.0 / 32.0) * input.cell(0, j).*;
            const right: f64 = (11.0 / 32.0) * input.cell(1, j).*;
            const first = left + right;
            const second = left - right;
            self.raw_finite = self.raw_finite and std.math.isFinite(left) and std.math.isFinite(right) and std.math.isFinite(first) and std.math.isFinite(second);
            if (j == 0) {
                self.raw_first = first;
                self.raw_second = second;
            }
            output.cell(0, j).* = first;
            output.cell(1, j).* = second;
        }
        self.returned_status = 0;
        return 0;
    }
};

test "native spectral enclosed dyadic symmetric action reports shifted recurrence overflow after finite callback" {
    var fixture: Fixture = .{};
    try fixture.init(1);
    defer fixture.deinit();
    var scratch = try SpectralSpan.init(&fixture, 18);
    defer scratch.deinit(&fixture);
    var probe: SpectralDyadicShiftProbe = .{ .owner = std.Thread.getCurrentId() };
    const input = [_]f64{ 1.4e308, 1.4e308, 1.4e308, 1.4e308 };
    const coefficients = [_]f64{ 1, 1 };
    const sentinel_nan: f64 = @bitCast(@as(u64, 0x7ff8000000000372));
    var output = [_]f64{ 773, 773, sentinel_nan, -0.0, 773, 773, sentinel_nan, -0.0, 991, 991 };
    const saved = output;
    var report: spectral.Report = .{};
    // H=(11/32)*[[1,1],[1,-1]], H^2=(121/512)I, so the
    // exact spectral radius is 11*sqrt(2)/32 < 1/2. The enclosure is valid.
    const radius = (@as(f128, 11) * @sqrt(@as(f128, 2))) / 32;
    try std.testing.expect(radius < @as(f128, 0.5));
    // Hx=(9.625e307,0) is finite; interval scaling gives
    // T1=(1.925e308,0), whose first scalar exceeds f64 MAX.
    const expected_raw = (@as(f128, 11) / 32) * input[0] + (@as(f128, 11) / 32) * input[1];
    try std.testing.expect(expected_raw < @as(f128, std.math.floatMax(f64)));
    try std.testing.expect(2 * expected_raw > @as(f128, std.math.floatMax(f64)));
    const request: spectral.Args = .{
        .kind = .series,
        .rows = 2,
        .columns = 2,
        .orders = 2,
        .lower = -0.5,
        .upper = 0.5,
        .input = spectralBuffer(&input, 2),
        .coefficients = spectralBuffer(&coefficients, 2),
        .output = spectralBuffer(&output, 4),
        .work = spectralBuffer(scratch.values, 2),
        .operator = .{ .apply = SpectralDyadicShiftProbe.apply, .data = &probe },
    };
    try std.testing.expectEqual(spectral.Status.numerical_failure, spectral.invoke(&fixture.context, request, &report));
    try std.testing.expect(probe.raw_finite);
    try std.testing.expectApproxEqRel(@as(f64, @floatCast(expected_raw)), probe.raw_first, 1e-15);
    try std.testing.expectEqual(@as(f64, 0), probe.raw_second);
    try std.testing.expect(2 * @as(f128, probe.raw_first) > @as(f128, std.math.floatMax(f64)));
    try std.testing.expectEqual(@as(i32, 0), probe.returned_status);
    try std.testing.expectEqual(@as(u64, 1), probe.calls);
    // c0*x was finite; the initial sum is intact because the first shift failed.
    try std.testing.expectEqual(input[0], scratch.values[12]);
    try std.testing.expectEqual(spectral.Phase.recurrence, report.phase);
    try std.testing.expect(report.admission_attempted and report.admission_succeeded and !report.output_complete);
    try std.testing.expect(report.progress.order_valid and report.progress.order == 1);
    try std.testing.expect(report.progress.callback_returned and report.progress.callback_order == 1);
    try std.testing.expectEqual(@as(i32, 0), report.progress.callback_status);
    try std.testing.expectEqual(@as(u64, 1), report.progress.callbacks_returned);
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&saved), std.mem.asBytes(&output));
    try std.testing.expectEqual(@as(usize, 0), fixture.context.snapshot().active_operations);
    try std.testing.expectEqual(@as(usize, 0), submissions);
}
