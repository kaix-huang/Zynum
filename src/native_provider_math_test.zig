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
