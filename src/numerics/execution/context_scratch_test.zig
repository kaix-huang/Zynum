// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");
const common = @import("context.zig");
const fixture = @import("test_support.zig");
const Context = common.Context;
const ReferenceContext = @import("legacy_adapter.zig").ReferenceContext;
fn progress(ctx: *Context) common.CloseResult {
    var report: @import("results.zig").CallReport = .{};
    return ctx.tryProgressClose(&report);
}
fn executor(ctx: *Context) *@import("owner.zig").Executor {
    const wrapper: *ReferenceContext = @fieldParentPtr("context", ctx);
    return &wrapper.executor;
}
const ContextScratch = common.ContextScratch;

/// Real allocator vtable backed by the existing counted c_allocator fixture.
/// Only the chosen lease size is observed; Executor allocations use this same
/// allocator but are not themselves claimed to be ContextScratch resources.
const ProbeAllocator = struct {
    backing: fixture.Allocations = .{ .budget = 100 },
    ctx: *Context = undefined,
    watch_bytes: usize,
    pause_free: std.atomic.Value(bool) = .init(false),
    free_entered: std.atomic.Value(bool) = .init(false),
    allow_free: std.atomic.Value(bool) = .init(false),
    free_completed: std.atomic.Value(bool) = .init(false),
    allocation_observed: std.atomic.Value(bool) = .init(false),
    close_during_free: common.CloseResult = undefined,

    fn allocator(self: *@This()) std.mem.Allocator {
        return .{ .ptr = self, .vtable = &.{ .alloc = alloc, .resize = resize, .remap = remap, .free = free } };
    }
    fn cast(raw: *anyopaque) *@This() {
        return @ptrCast(@alignCast(raw));
    }
    fn alloc(raw: *anyopaque, n: usize, a: std.mem.Alignment, ret: usize) ?[*]u8 {
        const self = cast(raw);
        if (n == self.watch_bytes) {
            const snapshot = self.ctx.snapshot();
            if (snapshot.resource_pins != 1 or snapshot.active_operations != 1)
                @panic("managed allocation must begin after pin, within its live operation");
            if (progress(self.ctx) != .busy) @panic("allocation callback cannot close its active operation");
            self.allocation_observed.store(true, .release);
        }
        return self.backing.allocator().rawAlloc(n, a, ret);
    }
    fn resize(_: *anyopaque, _: []u8, _: std.mem.Alignment, _: usize, _: usize) bool {
        return false;
    }
    fn remap(_: *anyopaque, _: []u8, _: std.mem.Alignment, _: usize, _: usize) ?[*]u8 {
        return null;
    }
    fn free(raw: *anyopaque, memory: []u8, a: std.mem.Alignment, ret: usize) void {
        const self = cast(raw);
        if (memory.len == self.watch_bytes) {
            if (self.ctx.snapshot().resource_pins != 1) @panic("managed free entered without its pin");
            self.close_during_free = progress(self.ctx);
            if (self.close_during_free == .closed) @panic("close passed a live managed free");
            self.free_entered.store(true, .release);
            if (self.pause_free.load(.acquire)) fixture.wait(&self.allow_free);
            self.backing.allocator().rawFree(memory, a, ret);
            if (self.ctx.snapshot().resource_pins != 1) @panic("pin released before actual free returned");
            self.free_completed.store(true, .release);
        } else self.backing.allocator().rawFree(memory, a, ret);
    }
};

fn finish(ctx: *Context) !void {
    _ = ctx.requestClose();
    try std.testing.expectEqual(common.CloseResult.closed, progress(ctx));
    try std.testing.expectEqual(@as(usize, 0), ctx.snapshot().resource_pins);
}

test "ContextScratch captures original allocator and pins before real allocation" {
    var probe: ProbeAllocator = .{ .watch_bytes = 6 * @sizeOf(f64) };
    var binding = probe.allocator();
    var reference_ctx: ReferenceContext = undefined;
    reference_ctx.initInPlace(binding, fixture.policy, .owned_closable, null);
    const ctx = reference_ctx.borrowedContext();
    probe.ctx = ctx;
    var op = try ctx.begin();
    var scratch = try ContextScratch(f64).acquire(&op, 6);
    var replacement: fixture.Allocations = .{ .budget = 0 };
    binding = replacement.allocator();
    try std.testing.expect(scratch.lease.allocator.ptr == probe.allocator().ptr);
    try std.testing.expect(scratch.lease.allocator.ptr != binding.ptr);
    try std.testing.expect(probe.allocation_observed.load(.acquire));
    try std.testing.expectEqual(@as(usize, 0), @intFromPtr(scratch.lease.values.ptr) % @alignOf(f64));
    for (scratch.lease.values, 0..) |*v, i| v.* = @as(f64, @floatFromInt(i)) + 0.25;
    for (scratch.lease.values, 0..) |v, i| try std.testing.expectEqual(@as(f64, @floatFromInt(i)) + 0.25, v);
    scratch.deinit();
    try std.testing.expect(probe.free_completed.load(.acquire));
    try std.testing.expectEqual(@as(usize, 0), ctx.snapshot().resource_pins);
    try std.testing.expectEqual(@as(usize, 1), ctx.snapshot().active_operations);
    try std.testing.expectEqual(@as(usize, 1), probe.backing.stats().freed);
    op.end();
    try finish(ctx);
    try std.testing.expectEqual(fixture.Allocations.Stats{
        .attempts = 0,
        .accepted = 0,
        .rejected = 0,
        .freed = 0,
    }, replacement.stats());
}

test "ContextScratch real allocation rejection rolls back pin without ending operation" {
    var probe: ProbeAllocator = .{ .watch_bytes = 47, .backing = .{ .budget = 0 } };
    var reference_ctx: ReferenceContext = undefined;
    reference_ctx.initInPlace(probe.allocator(), fixture.policy, .owned_closable, null);
    const ctx = reference_ctx.borrowedContext();
    probe.ctx = ctx;
    var op = try ctx.begin();
    _ = ctx.requestClose();
    try std.testing.expectError(error.OutOfMemory, ContextScratch(u8).acquire(&op, 47));
    try std.testing.expect(probe.allocation_observed.load(.acquire));
    try std.testing.expectEqual(@as(usize, 1), probe.backing.stats().rejected);
    try std.testing.expectEqual(@as(usize, 0), probe.backing.stats().freed);
    try std.testing.expectEqual(@as(usize, 0), ctx.snapshot().resource_pins);
    try std.testing.expectEqual(@as(usize, 1), ctx.snapshot().active_operations);
    try std.testing.expectEqual(common.CloseResult.busy, progress(ctx));
    op.end();
    try finish(ctx);
}

test "ContextScratch byte extent overflow rolls back pin before allocator entry" {
    var spy: fixture.Allocations = .{ .budget = 0 };
    var reference_ctx: ReferenceContext = undefined;
    reference_ctx.initInPlace(spy.allocator(), fixture.policy, .owned_closable, null);
    const ctx = reference_ctx.borrowedContext();
    var op = try ctx.begin();
    try std.testing.expectError(error.OutOfMemory, ContextScratch(f64).acquire(&op, std.math.maxInt(usize)));
    try std.testing.expectEqual(@as(usize, 0), spy.stats().attempts);
    try std.testing.expectEqual(@as(usize, 0), ctx.snapshot().resource_pins);
    op.end();
    try finish(ctx);
}

test "ContextScratch resource counter overflow refuses positive and zero counts before allocation" {
    var spy: fixture.Allocations = .{ .budget = 100 };
    var reference_ctx: ReferenceContext = undefined;
    reference_ctx.initInPlace(spy.allocator(), fixture.policy, .owned_closable, null);
    const ctx = reference_ctx.borrowedContext();
    Context.testing.configure(ctx, 1, null);
    var op = try ctx.begin();
    var existing_pin = try op.pinResource();
    try std.testing.expectError(error.CounterOverflow, ContextScratch(u8).acquire(&op, 13));
    try std.testing.expectError(error.CounterOverflow, ContextScratch(u8).acquire(&op, 0));
    try std.testing.expectEqual(@as(usize, 0), spy.stats().attempts);
    try std.testing.expectEqual(@as(usize, 1), ctx.snapshot().resource_pins);
    existing_pin.releaseAfterResourceFree();
    op.end();
    try finish(ctx);
}

test "ContextScratch zero length and zero-sized elements preserve allocator semantics and lifetime" {
    var spy: fixture.Allocations = .{ .budget = 0 };
    var reference_ctx: ReferenceContext = undefined;
    reference_ctx.initInPlace(spy.allocator(), fixture.policy, .owned_closable, null);
    const ctx = reference_ctx.borrowedContext();
    var op = try ctx.begin();
    var empty = try ContextScratch(u64).acquire(&op, 0);
    var zero_size = try ContextScratch(struct {}).acquire(&op, 7);
    try std.testing.expectEqual(@as(usize, 0), empty.lease.values.len);
    try std.testing.expectEqual(@as(usize, 7), zero_size.lease.values.len);
    try std.testing.expectEqual(@as(usize, 0), spy.stats().attempts);
    try std.testing.expectEqual(@as(usize, 2), ctx.snapshot().resource_pins);
    op.end();
    _ = ctx.requestClose();
    try std.testing.expectEqual(common.CloseResult.waiting_resources, progress(ctx));
    empty.deinit();
    try std.testing.expectEqual(common.CloseResult.waiting_resources, progress(ctx));
    zero_size.deinit();
    try std.testing.expectEqual(@as(usize, 0), spy.stats().freed);
    try finish(ctx);
}

test "ContextScratch acquired after close request survives operation end until lease release" {
    var probe: ProbeAllocator = .{ .watch_bytes = 41 };
    var reference_ctx: ReferenceContext = undefined;
    reference_ctx.initInPlace(probe.allocator(), fixture.policy, .owned_closable, null);
    const ctx = reference_ctx.borrowedContext();
    probe.ctx = ctx;
    var op = try ctx.begin();
    _ = ctx.requestClose();
    var scratch = try ContextScratch(u8).acquire(&op, 41);
    @memset(scratch.lease.values, 0xa7);
    op.end();
    try std.testing.expectEqual(common.CloseResult.waiting_resources, progress(ctx));
    try std.testing.expectError(error.Closing, ctx.begin());
    for (scratch.lease.values) |v| try std.testing.expectEqual(@as(u8, 0xa7), v);
    scratch.deinit();
    try std.testing.expectEqual(common.CloseResult.waiting_resources, probe.close_during_free);
    try finish(ctx);
}

test "ContextScratch paused real free and callback reentry cannot publish closed" {
    var probe: ProbeAllocator = .{ .watch_bytes = 43, .pause_free = .init(true) };
    var reference_ctx: ReferenceContext = undefined;
    reference_ctx.initInPlace(probe.allocator(), fixture.policy, .owned_closable, null);
    const ctx = reference_ctx.borrowedContext();
    probe.ctx = ctx;
    var op = try ctx.begin();
    var scratch = try ContextScratch(u8).acquire(&op, 43);
    op.end();
    _ = ctx.requestClose();
    const Cleanup = struct {
        fn run(lease: *ContextScratch(u8)) void {
            lease.deinit();
        }
    };
    const thread = try std.Thread.spawn(.{}, Cleanup.run, .{&scratch});
    var joined = false;
    defer {
        probe.allow_free.store(true, .release);
        if (!joined) thread.join();
    }
    fixture.wait(&probe.free_entered);
    try std.testing.expectEqual(common.CloseResult.waiting_resources, progress(ctx));
    try std.testing.expectEqual(common.Phase.drained, ctx.snapshot().phase);
    try std.testing.expectEqual(@as(usize, 1), ctx.snapshot().resource_pins);
    try std.testing.expect(!probe.free_completed.load(.acquire));
    try std.testing.expectEqual(@as(usize, 0), probe.backing.stats().freed);
    probe.allow_free.store(true, .release);
    thread.join();
    joined = true;
    try std.testing.expect(probe.free_completed.load(.acquire));
    try std.testing.expectEqual(@as(usize, 1), probe.backing.stats().freed);
    try finish(ctx);
}

const WorkerResource = struct {
    ctx: *Context,
    scratch: ContextScratch(u8),
    cleaned: *std.atomic.Value(bool),
};
threadlocal var worker_resource: ?WorkerResource = null;
fn releaseWorkerResource() void {
    if (worker_resource) |*resource| {
        if (progress(resource.ctx) != .busy) @panic("worker cleanup must not reenter finalizer");
        for (resource.scratch.lease.values) |v| if (v != 0xa5) @panic("worker scratch contents changed");
        resource.scratch.deinit();
        resource.cleaned.store(true, .release);
        worker_resource = null;
    }
}

test "ContextScratch worker-held last lease is released by drain before closed" {
    fixture.setThreads(2);
    var probe: ProbeAllocator = .{ .watch_bytes = 37, .pause_free = .init(true) };
    var reference_ctx: ReferenceContext = undefined;
    reference_ctx.initInPlace(probe.allocator(), fixture.policy, .owned_closable, null);
    const ctx = reference_ctx.borrowedContext();
    probe.ctx = ctx;
    defer executor(ctx).drainAndReset();
    var op = try ctx.begin();
    const Work = struct {
        op: *const common.Operation,
        cleaned: std.atomic.Value(bool) = .init(false),
        fn task(raw: *const anyopaque, i: usize) void {
            const self: *@This() = @ptrCast(@alignCast(@constCast(raw)));
            if (i == 0) return;
            const scratch = ContextScratch(u8).acquire(self.op, 37) catch @panic("scratch fixture");
            @memset(scratch.lease.values, 0xa5);
            worker_resource = .{ .ctx = self.op.borrowedContext(), .scratch = scratch, .cleaned = &self.cleaned };
        }
    };
    var work: Work = .{ .op = &op };
    const domain: @import("owner.zig").WorkerCacheDomain = .{ .release_current_worker = releaseWorkerResource };
    try std.testing.expect(executor(ctx).runLowLatency(.{ .task_fn = Work.task, .tasks = &work, .count = 2, .cache_domain = &domain }));
    try std.testing.expectEqual(@as(usize, 1), ctx.snapshot().resource_pins);
    try std.testing.expect(!work.cleaned.load(.acquire));
    op.end();
    _ = ctx.requestClose();
    const Finalize = struct {
        ctx: *Context,
        result: common.CloseResult = undefined,
        done: std.atomic.Value(bool) = .init(false),
        fn run(self: *@This()) void {
            self.result = progress(self.ctx);
            self.done.store(true, .release);
        }
    };
    var finalizer: Finalize = .{ .ctx = ctx };
    const thread = try std.Thread.spawn(.{}, Finalize.run, .{&finalizer});
    var joined = false;
    defer {
        probe.allow_free.store(true, .release);
        if (!joined) thread.join();
    }
    fixture.wait(&probe.free_entered);
    try std.testing.expectEqual(common.Phase.draining, ctx.snapshot().phase);
    try std.testing.expectEqual(common.CloseResult.busy, progress(ctx));
    try std.testing.expectEqual(@as(usize, 1), ctx.snapshot().resource_pins);
    try std.testing.expect(!finalizer.done.load(.acquire));
    try std.testing.expect(!work.cleaned.load(.acquire));
    probe.allow_free.store(true, .release);
    thread.join();
    joined = true;
    try std.testing.expectEqual(common.CloseResult.closed, finalizer.result);
    try std.testing.expect(work.cleaned.load(.acquire));
    try std.testing.expect(probe.free_completed.load(.acquire));
    try std.testing.expectEqual(common.CloseResult.busy, probe.close_during_free);
    try std.testing.expectEqual(@as(usize, 0), ctx.snapshot().resource_pins);
    try std.testing.expectEqual(probe.backing.stats().accepted, probe.backing.stats().freed);
}
