// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Synchronous host services for the matched structured object. The opaque
//! binding borrows the incoming operation; it owns no Context or executor.

const std = @import("std");
const abi = @import("x86_64_structured_abi.zig");
const core_pool = @import("../../core/execution/thread_pool.zig");
const gemm_host = @import("structured_gemm_host.zig");

pub const Binding = struct {
    resources: core_pool.CallResources,
    captured_allocator: std.mem.Allocator,

    pub fn init(resources: core_pool.CallResources) Binding {
        return .{
            .resources = resources,
            .captured_allocator = switch (resources) {
                .admitted => |operation| operation.allocator(),
                .legacy_default => std.heap.c_allocator,
            },
        };
    }

    /// Construct only after this Binding has its final address. The descriptor
    /// and this binding live until execute_v2 and all synchronous callbacks end.
    pub fn descriptor(self: *const Binding) abi.ServicesV1 {
        return .{
            .version = 1,
            .struct_bytes = @sizeOf(abi.ServicesV1),
            .borrow = self,
            .alloc = allocate,
            .free = release,
            .task_count = taskCount,
            .run_low_latency = runLowLatency,
            .gemm = gemm,
        };
    }
};

fn binding(raw: *const anyopaque) *const Binding {
    return @ptrCast(@alignCast(raw));
}

// std.mem.Alignment is a log2 enum, not a byte alignment. Range checking before
// narrowing also keeps the conversion valid for a non-x86 host import.
fn alignment(log2: u8) ?std.mem.Alignment {
    if (log2 >= @bitSizeOf(usize)) return null;
    return @fromBackingInt(@as(std.math.Log2Int(usize), @intCast(log2)));
}

fn allocate(raw: *const anyopaque, bytes: usize, alignment_log2: u8) callconv(.c) ?[*]u8 {
    if (bytes == 0) return null;
    const requested = alignment(alignment_log2) orelse return null;
    return binding(raw).captured_allocator.rawAlloc(bytes, requested, @returnAddress());
}

fn release(raw: *const anyopaque, memory: [*]u8, bytes: usize, alignment_log2: u8) callconv(.c) void {
    // The matched object must return the original successful allocation tuple;
    // invalid free metadata is a protocol bug, never a default-allocator retry.
    const requested = alignment(alignment_log2) orelse unreachable;
    std.debug.assert(bytes != 0);
    binding(raw).captured_allocator.rawFree(memory[0..bytes], requested, @returnAddress());
}

fn taskCount(raw: *const anyopaque, items: usize, min_items: usize) callconv(.c) usize {
    return binding(raw).resources.taskCount(items, min_items);
}

const TaskBorrow = struct {
    task_fn: core_pool.TaskFn,
    tasks: *const anyopaque,
};

fn runTask(raw: *const anyopaque, index: usize) void {
    const task: *const TaskBorrow = @ptrCast(@alignCast(raw));
    task.task_fn(task.tasks, index);
}

fn runLowLatency(raw: *const anyopaque, task_address: usize, tasks: *const anyopaque, count: usize) callconv(.c) u8 {
    // Only this service uses the C convention. Reconstruct the original Zig
    // task convention, retaining the existing joined/refused-before-work rule.
    const task: TaskBorrow = .{ .task_fn = @ptrFromInt(task_address), .tasks = tasks };
    return @intFromBool(binding(raw).resources.runLowLatency(runTask, &task, count));
}

fn gemm(raw: *const anyopaque, request: *abi.GemmRequest) callconv(.c) void {
    gemm_host.execute(binding(raw).resources, request);
}

test "structured service callbacks borrow allocator executor and closing GEMM operation" {
    const execution = @import("zynum-execution");
    const fixture = execution.test_support;
    fixture.setThreads(4);
    var allocations: fixture.Allocations = .{ .budget = 1 };
    const ReferenceFixture = struct {
        const Context = @import("zynum-execution").reference.ReferenceContext;
        fn progressClose(ctx: *@import("zynum-execution").Context) @import("zynum-execution").CloseResult {
            var report: @import("zynum-execution").CallReport = .{};
            return ctx.tryProgressClose(&report);
        }
    };
    var reference_ctx: ReferenceFixture.Context = undefined;
    reference_ctx.initInPlace(allocations.allocator(), fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const ctx = reference_ctx.borrowedContext();
    defer {
        _ = ctx.requestClose();
        _ = ReferenceFixture.progressClose(ctx);
    }
    var operation = try ctx.begin();
    var ended = false;
    defer if (!ended) operation.end();
    const host = Binding.init(.{ .admitted = &operation });
    const services = host.descriptor();
    try std.testing.expect(abi.servicesSupported(&services));
    try std.testing.expectEqual(@as(usize, 4), services.task_count(services.borrow, 64, 1));
    try std.testing.expect(services.alloc(services.borrow, 0, 6) == null);
    try std.testing.expect(services.alloc(services.borrow, 64, 255) == null);
    try std.testing.expectEqual(@as(usize, 0), allocations.stats().attempts);
    const memory = services.alloc(services.borrow, 128, 6) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(usize, 0), @intFromPtr(memory) % 64);
    @memset(memory[0..128], 0xa5);
    try std.testing.expect(services.alloc(services.borrow, 64, 6) == null);
    services.free(services.borrow, memory, 128, 6);
    try std.testing.expectEqual(@as(usize, 1), allocations.stats().accepted);
    try std.testing.expectEqual(@as(usize, 1), allocations.stats().freed);
    try std.testing.expectEqual(@as(usize, 1), allocations.stats().rejected);
    allocations.budget = std.math.maxInt(usize);
    const Tasks = struct {
        hits: [4]std.atomic.Value(usize) = @splat(.init(0)),
        fn run(raw: *const anyopaque, index: usize) void {
            const self: *@This() = @ptrCast(@alignCast(@constCast(raw)));
            _ = self.hits[index].fetchAdd(1, .monotonic);
        }
    };
    var tasks: Tasks = .{};
    const task_fn: core_pool.TaskFn = Tasks.run;
    try std.testing.expectEqual(@as(u8, 1), services.run_low_latency(services.borrow, @intFromPtr(task_fn), &tasks, 4));
    for (&tasks.hits) |*hits| try std.testing.expectEqual(@as(usize, 1), hits.load(.acquire));
    try std.testing.expectEqual(execution.context.RequestResult.pending, ctx.requestClose());
    try std.testing.expectEqual(execution.context.CloseResult.busy, ReferenceFixture.progressClose(ctx));
    try std.testing.expectError(error.Closing, ctx.begin());
    try std.testing.expectEqual(@as(u8, 1), services.run_low_latency(services.borrow, @intFromPtr(task_fn), &tasks, 4));
    for (&tasks.hits) |*hits| try std.testing.expectEqual(@as(usize, 2), hits.load(.acquire));
    const a = [_]f64{ 1, 2, 3, 4 };
    const b = [_]f64{ 5, 6, 7, 8 };
    var c: [4]f64 = @splat(std.math.nan(f64));
    var request: abi.GemmRequest = .{
        .transa = @backingInt(abi.Transpose.no_trans),
        .transb = @backingInt(abi.Transpose.no_trans),
        .scalar = @backingInt(abi.Scalar.f64),
        .m = 2,
        .n = 2,
        .k = 2,
        .lda = 2,
        .ldb = 2,
        .ldc = 2,
        .alpha_re = @bitCast(@as(f64, 1)),
        .alpha_im = 0,
        .beta_re = 0,
        .beta_im = 0,
        .a = &a,
        .b = &b,
        .c = &c,
    };
    services.gemm(services.borrow, &request);
    try std.testing.expectEqualSlices(f64, &.{ 23, 34, 31, 46 }, &c);
    try std.testing.expectEqual(@as(usize, 1), ctx.snapshot().active_operations);
    operation.end();
    ended = true;
    try std.testing.expectEqual(execution.context.CloseResult.closed, ReferenceFixture.progressClose(ctx));
    try std.testing.expectEqual(allocations.stats().accepted, allocations.stats().freed);
}
