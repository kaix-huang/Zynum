// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");
const common = @import("context.zig");
const fixture = @import("test_support.zig");
const Context = common.Context;

fn finish(ctx: *Context) !void {
    try std.testing.expectEqual(common.RequestResult.pending, ctx.requestClose());
    try std.testing.expectEqual(common.CloseResult.closed, ctx.tryProgressClose());
    try std.testing.expectEqual(common.CloseResult.closed, ctx.tryProgressClose());
    try std.testing.expectEqual(common.RequestResult.closed, ctx.requestClose());
}

test "Context admission and pins allocate nothing and immortal reset stays restartable" {
    fixture.setThreads(1);
    var spy: fixture.Allocations = .{ .budget = 0 };
    var ctx = Context.init(spy.allocator(), fixture.policy, .immortal_default);
    var operation = try ctx.begin();
    var pin = try operation.pinResource();
    try std.testing.expect(operation.allocator().ptr == spy.allocator().ptr);
    try std.testing.expect(ctx.executor.allocator.ptr == ctx.allocator.ptr);
    try std.testing.expectEqual(common.RequestResult.immortal, ctx.requestClose());
    try std.testing.expectEqual(common.CloseResult.immortal, ctx.tryProgressClose());
    pin.releaseAfterResourceFree();
    operation.end();
    ctx.executor.drainAndReset();
    var again = try ctx.begin();
    again.end();
    try std.testing.expectEqual(common.Phase.open, ctx.snapshot().phase);
    try std.testing.expectEqual(@as(usize, 0), ctx.snapshot().active_operations);
    try std.testing.expectEqual(@as(usize, 0), spy.stats().attempts);
}

test "Context checked counter limit rejects without wrapping or losing existing lifetimes" {
    var ctx = Context.init(std.heap.c_allocator, fixture.policy, .internal_closable);
    Context.testing.configure(&ctx, 1, null);
    var op = try ctx.begin();
    try std.testing.expectError(error.CounterOverflow, ctx.begin());
    var pin = try op.pinResource();
    try std.testing.expectError(error.CounterOverflow, op.pinResource());
    try std.testing.expectEqual(@as(usize, 1), ctx.snapshot().active_operations);
    try std.testing.expectEqual(@as(usize, 1), ctx.snapshot().resource_pins);
    _ = ctx.requestClose();
    try std.testing.expectError(error.Closing, ctx.begin());
    try std.testing.expectEqual(common.CloseResult.busy, ctx.tryProgressClose());
    op.end();
    try std.testing.expectEqual(common.CloseResult.waiting_resources, ctx.tryProgressClose());
    pin.releaseAfterResourceFree();
    try std.testing.expectEqual(common.CloseResult.closed, ctx.tryProgressClose());
}

test "Context admitted operation may acquire its cleanup pin after close request" {
    var ctx = Context.init(std.heap.c_allocator, fixture.policy, .internal_closable);
    var op = try ctx.begin();
    _ = ctx.requestClose();
    var pin = try op.pinResource();
    op.end();
    try std.testing.expectEqual(common.CloseResult.waiting_resources, ctx.tryProgressClose());
    pin.releaseAfterResourceFree();
    try std.testing.expectEqual(common.CloseResult.closed, ctx.tryProgressClose());
}

fn admittedScale(ctx: *Context, values: []f64) !void {
    var op = try ctx.begin();
    defer op.end();
    for (values) |*v| v.* = 3 * v.* - 2;
}
test "Context close before admission refuses the real body without output mutation" {
    var ctx = Context.init(std.heap.c_allocator, fixture.policy, .internal_closable);
    var values = [_]f64{ 1, 2, 3 };
    _ = ctx.requestClose();
    try std.testing.expectError(error.Closing, admittedScale(&ctx, &values));
    try std.testing.expectEqualSlices(f64, &.{ 1, 2, 3 }, &values);
    try std.testing.expectEqual(common.CloseResult.closed, ctx.tryProgressClose());
}

test "Context simultaneous callers keep executor refusal numeric fallback alive through close" {
    fixture.setThreads(2);
    var ctx = Context.init(std.heap.c_allocator, fixture.policy, .internal_closable);
    const Work = struct {
        ctx: *Context,
        entered: std.atomic.Value(bool) = .init(false),
        release: std.atomic.Value(bool) = .init(false),
        finished: std.atomic.Value(bool) = .init(false),
        visits: [2]std.atomic.Value(u32) = @splat(.init(0)),
        accepted: bool = false,
        fn task(raw: *const anyopaque, i: usize) void {
            const self: *@This() = @ptrCast(@alignCast(@constCast(raw)));
            _ = self.visits[i].fetchAdd(1, .monotonic);
            if (i == 0) {
                self.entered.store(true, .release);
                fixture.wait(&self.release);
            }
        }
        fn run(self: *@This()) void {
            defer self.finished.store(true, .release);
            var op = self.ctx.begin() catch @panic("fixture begin");
            defer op.end();
            self.accepted = self.ctx.executor.runLowLatency(.{ .task_fn = task, .tasks = self, .count = 2 });
        }
    };
    const Fallback = struct {
        ctx: *Context,
        body_done: std.atomic.Value(bool) = .init(false),
        release: std.atomic.Value(bool) = .init(false),
        values: [64]f64 = undefined,
        refused: bool = false,
        fn forbidden(_: *const anyopaque, _: usize) void {
            @panic("refused batch executed");
        }
        fn run(self: *@This()) void {
            var op = self.ctx.begin() catch @panic("fixture begin");
            defer op.end();
            self.refused = !self.ctx.executor.run(.{ .task_fn = forbidden, .tasks = self, .count = 2 });
            if (!self.refused) @panic("busy executor did not refuse");
            // An actual numeric body remains within the same operation after refusal.
            for (&self.values, 0..) |*v, i| v.* = 3 * @as(f64, @floatFromInt(i)) - 2;
            self.body_done.store(true, .release);
            fixture.wait(&self.release);
        }
    };
    var work: Work = .{ .ctx = &ctx };
    const primary = try std.Thread.spawn(.{}, Work.run, .{&work});
    var primary_joined = false;
    defer {
        work.release.store(true, .release);
        if (!primary_joined) primary.join();
        ctx.executor.drainAndReset();
    }
    while (!work.entered.load(.acquire) and !work.finished.load(.acquire)) std.Thread.yield() catch {};
    try std.testing.expect(work.entered.load(.acquire));
    var fallback: Fallback = .{ .ctx = &ctx };
    const second = try std.Thread.spawn(.{}, Fallback.run, .{&fallback});
    var second_joined = false;
    defer {
        fallback.release.store(true, .release);
        if (!second_joined) second.join();
    }
    fixture.wait(&fallback.body_done);
    try std.testing.expectEqual(@as(usize, 2), ctx.snapshot().active_operations);
    _ = ctx.requestClose();
    try std.testing.expectEqual(common.CloseResult.busy, ctx.tryProgressClose());
    for (fallback.values, 0..) |v, i| try std.testing.expectEqual(3 * @as(f64, @floatFromInt(i)) - 2, v);
    work.release.store(true, .release);
    primary.join();
    primary_joined = true;
    try std.testing.expect(work.accepted and fallback.refused);
    try std.testing.expectEqual(@as(usize, 1), ctx.snapshot().active_operations);
    try std.testing.expectEqual(common.CloseResult.busy, ctx.tryProgressClose());
    fallback.release.store(true, .release);
    second.join();
    second_joined = true;
    for (&work.visits) |*v| try std.testing.expectEqual(@as(u32, 1), v.load(.acquire));
    try std.testing.expectEqual(common.CloseResult.closed, ctx.tryProgressClose());
}

test "Context callback close is Busy without waiting for its own real operation" {
    const Hooks = struct {
        fn observe(ctx: *Context, _: common.TestEvent) void {
            if (ctx.requestClose() != .pending or ctx.tryProgressClose() != .busy)
                @panic("callback close must not wait or finish its own operation");
        }
    };
    var ctx = Context.init(std.heap.c_allocator, fixture.policy, .internal_closable);
    Context.testing.configure(&ctx, std.math.maxInt(usize), Hooks.observe);
    var values = [_]f64{ 1, 2, 3 };
    try admittedScale(&ctx, &values);
    try std.testing.expectEqualSlices(f64, &.{ 1, 4, 7 }, &values);
    try std.testing.expectEqual(common.CloseResult.closed, ctx.tryProgressClose());
}

const FreeBarrier = struct {
    ctx: *Context = undefined,
    entered: std.atomic.Value(bool) = .init(false),
    release: std.atomic.Value(bool) = .init(false),
    freed: std.atomic.Value(bool) = .init(false),
    fn allocator(self: *@This()) std.mem.Allocator {
        return .{ .ptr = self, .vtable = &.{ .alloc = alloc, .resize = resize, .remap = remap, .free = free } };
    }
    fn alloc(_: *anyopaque, n: usize, a: std.mem.Alignment, ret: usize) ?[*]u8 {
        return std.heap.c_allocator.rawAlloc(n, a, ret);
    }
    fn resize(_: *anyopaque, _: []u8, _: std.mem.Alignment, _: usize, _: usize) bool {
        return false;
    }
    fn remap(_: *anyopaque, _: []u8, _: std.mem.Alignment, _: usize, _: usize) ?[*]u8 {
        return null;
    }
    fn free(raw: *anyopaque, memory: []u8, a: std.mem.Alignment, ret: usize) void {
        const self: *@This() = @ptrCast(@alignCast(raw));
        if (self.ctx.snapshot().resource_pins != 1) @panic("pin released before actual free returned");
        // Calling back into close while free runs must not deadlock on gate.
        if (self.ctx.tryProgressClose() != .waiting_resources) @panic("allocator still in use");
        self.entered.store(true, .release);
        fixture.wait(&self.release);
        std.heap.c_allocator.rawFree(memory, a, ret);
        self.freed.store(true, .release);
    }
};
test "Context resource pin survives actual captured allocator free and cleanup reentry" {
    var spy: FreeBarrier = .{};
    var allocator_binding = spy.allocator();
    var ctx = Context.init(allocator_binding, fixture.policy, .internal_closable);
    spy.ctx = &ctx;
    var op = try ctx.begin();
    var pin = try op.pinResource();
    const captured = op.allocator();
    const memory = try captured.alloc(u8, 41);
    @memset(memory, 0xa7);
    var replacement: fixture.Allocations = .{ .budget = 0 };
    allocator_binding = replacement.allocator();
    try std.testing.expect(captured.ptr != allocator_binding.ptr);
    op.end();
    _ = ctx.requestClose();
    const Cleanup = struct {
        allocator: std.mem.Allocator,
        memory: []u8,
        pin: *common.ResourcePin,
        fn run(self: *@This()) void {
            self.allocator.free(self.memory);
            self.pin.releaseAfterResourceFree();
        }
    };
    var cleanup: Cleanup = .{ .allocator = captured, .memory = memory, .pin = &pin };
    const thread = try std.Thread.spawn(.{}, Cleanup.run, .{&cleanup});
    var joined = false;
    defer {
        spy.release.store(true, .release);
        if (!joined) thread.join();
    }
    fixture.wait(&spy.entered);
    try std.testing.expectEqual(common.CloseResult.waiting_resources, ctx.tryProgressClose());
    try std.testing.expect(!spy.freed.load(.acquire));
    spy.release.store(true, .release);
    thread.join();
    joined = true;
    try std.testing.expect(spy.freed.load(.acquire));
    try std.testing.expectEqual(common.CloseResult.closed, ctx.tryProgressClose());
    try std.testing.expectEqual(fixture.Allocations.Stats{
        .attempts = 0,
        .accepted = 0,
        .rejected = 0,
        .freed = 0,
    }, replacement.stats());
}

const WorkerResource = struct {
    ctx: *Context,
    pin: common.ResourcePin,
    allocator: std.mem.Allocator,
    memory: []u8,
    cleaned: *std.atomic.Value(bool),
};
threadlocal var worker_resource: ?WorkerResource = null;
fn releaseWorkerResource() void {
    if (worker_resource) |*resource| {
        if (resource.ctx.tryProgressClose() != .busy) @panic("drain callback must not reenter finalizer");
        resource.allocator.free(resource.memory);
        resource.pin.releaseAfterResourceFree();
        resource.cleaned.store(true, .release);
        worker_resource = null;
    }
}
test "Context drains persistent worker holding last resource pin before waiting on resources" {
    fixture.setThreads(2);
    var spy: fixture.Allocations = .{ .budget = 100 };
    var ctx = Context.init(spy.allocator(), fixture.policy, .internal_closable);
    defer ctx.executor.drainAndReset();
    var op = try ctx.begin();
    const Work = struct {
        op: *const common.Operation,
        cleaned: std.atomic.Value(bool) = .init(false),
        fn task(raw: *const anyopaque, i: usize) void {
            const self: *@This() = @ptrCast(@alignCast(@constCast(raw)));
            if (i == 0) return;
            const allocator = self.op.allocator();
            var pin = self.op.pinResource() catch @panic("pin fixture");
            const memory = allocator.alloc(u8, 37) catch {
                pin.releaseAfterResourceFree();
                @panic("allocation fixture");
            };
            @memset(memory, 0xa5);
            worker_resource = .{ .ctx = self.op.borrowedContext(), .pin = pin, .allocator = allocator, .memory = memory, .cleaned = &self.cleaned };
        }
    };
    var work: Work = .{ .op = &op };
    const domain: @import("owner.zig").WorkerCacheDomain = .{ .release_current_worker = releaseWorkerResource };
    try std.testing.expect(ctx.executor.runLowLatency(.{ .task_fn = Work.task, .tasks = &work, .count = 2, .cache_domain = &domain }));
    try std.testing.expectEqual(@as(u8, 2), ctx.executor.persistent_init_state.load(.acquire));
    try std.testing.expectEqual(@as(usize, 1), ctx.snapshot().resource_pins);
    op.end();
    try finish(&ctx);
    try std.testing.expect(work.cleaned.load(.acquire));
    try std.testing.expectEqual(spy.stats().accepted, spy.stats().freed);
}
