// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
const std = @import("std");
const common = @import("context.zig");
const resources = @import("call_resources.zig");
const results = @import("results.zig");
const c = @import("heteris-provider-c");
const Fake = struct {
    allocator: std.mem.Allocator,
    references: usize = 1,
    retains: usize = 0,
    retain_status: i32 = 0,
    releases: usize = 0,
    query_status: i32 = 0,
    release_status: i32 = 0,
    scratch_release_status: i32 = 0,
    flags: u64 = 31,
    budget: u64 = 65536,
    used: u64 = 0,
    next: u64 = 100,
    allocations: [64]?Allocation = @splat(null),
    bulk_calls: usize = 0,
    allocation_calls: usize = 0,
    pause_release: bool = false,
    release_entered: std.atomic.Value(bool) = .init(false),
    allow_release: std.atomic.Value(bool) = .init(false),
    behavior: enum { normal, reject, malformed_range, partial, malformed_completion, malformed_counts } = .normal,
    context: ?*common.Context = null,
    const Allocation = struct { serial: u64, pointer: [*]u8, bytes: usize, alignment: std.mem.Alignment, charge: u64 };
    fn token(self: *@This(), serial: u64, kind: u32) c.heteris_rp_token {
        return .{ .authority = self, .serial = serial, .generation = 1, .kind = kind, .reserved = 0 };
    }
    fn cast(t: c.heteris_rp_token) *@This() {
        return @ptrCast(@alignCast(t.authority.?));
    }
    fn retain(t: c.heteris_rp_token) callconv(.c) i32 {
        const self = cast(t);
        self.retains += 1;
        if (self.retain_status != 0) return self.retain_status;
        self.references += 1;
        return 0;
    }
    fn release(t: c.heteris_rp_token) callconv(.c) i32 {
        const self = cast(t);
        self.releases += 1;
        if (self.context) |ctx| {
            var report: results.CallReport = .{};
            if (ctx.tryProgressClose(&report) != .busy) @panic("provider release callback must run outside gate in releasing phase");
        }
        if (self.pause_release) {
            self.release_entered.store(true, .release);
            while (!self.allow_release.load(.acquire)) std.Thread.yield() catch {};
        }
        if (self.release_status != 0) return self.release_status;
        self.references -= 1;
        return 0;
    }
    fn query(t: c.heteris_rp_token, out: [*c]c.heteris_rp_cpu_caps_v1) callconv(.c) i32 {
        const self = cast(t);
        if (self.query_status != 0) return self.query_status;
        out[0] = .{ .abi_version = 1, .struct_size = @sizeOf(c.heteris_rp_cpu_caps_v1), .mode = c.HETERIS_RESOURCE_MODE_INTEGRATED, .device_kind = c.HETERIS_DEVICE_CPU, .flags = self.flags, .provider_instance = .{ .serial = 1, .generation = 1 }, .runtime_instance = .{ .serial = 2, .generation = 1 }, .memory_domain = .{ .serial = 3, .generation = 1 }, .compute_workers = 2, .managed_threads = 3, .max_bulk_commands = 2, .reserved = 0, .host_bytes_limit = self.budget, .task_scratch_bytes_limit = 0 };
        return 0;
    }
    fn same(_: c.heteris_rp_token, _: c.heteris_rp_token, out: [*c]u32) callconv(.c) i32 {
        out[0] = 1;
        return 0;
    }
    fn allocate(t: c.heteris_rp_token, bytes: u64, alignment: u64, out: [*c]c.heteris_rp_scratch_v1) callconv(.c) i32 {
        const self = cast(t);
        self.allocation_calls += 1;
        if (self.context) |ctx| if (ctx.snapshot().resource_pins == 0) @panic("scratch allocation must follow pin outside gate");
        const charge = std.math.add(u64, bytes, alignment - 1) catch return c.HETERIS_OVERFLOW;
        if (charge > self.budget - self.used) return c.HETERIS_BUDGET_EXCEEDED;
        const n = std.math.cast(usize, @max(bytes, 1)) orelse return c.HETERIS_OVERFLOW;
        const a = std.mem.Alignment.fromByteUnits(@intCast(alignment));
        const pointer = self.allocator.rawAlloc(n, a, @returnAddress()) orelse return c.HETERIS_OUT_OF_MEMORY;
        self.next += 1;
        for (&self.allocations) |*entry| if (entry.* == null) {
            entry.* = .{ .serial = self.next, .pointer = pointer, .bytes = n, .alignment = a, .charge = charge };
            self.used += charge;
            out[0] = .{ .abi_version = 1, .struct_size = @sizeOf(c.heteris_rp_scratch_v1), .lease = self.token(self.next, c.HETERIS_RP_SCRATCH), .memory_domain = .{ .serial = 3, .generation = 1 }, .data = pointer, .logical_bytes = bytes, .alignment = alignment, .accounted_bytes = charge };
            return 0;
        };
        @panic("fake allocation record capacity");
    }
    fn scratchRetain(_: c.heteris_rp_token) callconv(.c) i32 {
        return c.HETERIS_UNSUPPORTED;
    }
    fn scratchRelease(t: c.heteris_rp_token) callconv(.c) i32 {
        const self = cast(t);
        if (self.scratch_release_status != 0) return self.scratch_release_status;
        if (self.context) |ctx| if (ctx.snapshot().resource_pins == 0) @panic("scratch free lost its Context pin");
        for (&self.allocations) |*entry| if (entry.*) |allocation| {
            if (allocation.serial == t.serial) {
                self.allocator.rawFree(allocation.pointer[0..allocation.bytes], allocation.alignment, @returnAddress());
                self.used -= allocation.charge;
                entry.* = null;
                return 0;
            }
        };
        return c.HETERIS_INVALID_HANDLE;
    }
    fn bulk(t: c.heteris_rp_token, begin: u64, end: u64, _: u64, _: u32, callback: c.heteris_rp_cpu_range_fn, data: ?*anyopaque, out: [*c]c.heteris_rp_bulk_result_v1) callconv(.c) i32 {
        const self = cast(t);
        self.bulk_calls += 1;
        out[0] = .{ .abi_version = 1, .struct_size = @sizeOf(c.heteris_rp_bulk_result_v1), .state = c.HETERIS_RP_QUIESCENT, .reserved = 0, .execution_status = 0, .callback_status = 0, .callbacks_started = 0, .callbacks_completed = 0 };
        if (self.behavior == .reject) {
            out[0].state = c.HETERIS_RP_REJECTED;
            out[0].execution_status = -778;
            return -778;
        }
        if (self.behavior == .malformed_completion) {
            out[0].state = 999;
            return 0;
        }
        if (self.behavior == .malformed_counts) {
            out[0].execution_status = -778;
            out[0].callback_status = -1903;
            out[0].callbacks_started = end + 1;
            out[0].callbacks_completed = end + 1;
            return -778;
        }
        if (self.behavior == .malformed_range) {
            out[0].callback_status = callback.?(data, begin, end + 1);
            out[0].callbacks_started = 1;
            out[0].callbacks_completed = 1;
            return out[0].callback_status;
        }
        for (begin..end) |index| {
            const status = callback.?(data, index, index + 1);
            out[0].callbacks_started += 1;
            out[0].callbacks_completed += 1;
            if (status != 0) {
                out[0].callback_status = status;
                return status;
            }
            if (self.behavior == .partial) {
                out[0].callback_status = -1903;
                return -1903;
            }
        }
        return 0;
    }
    fn api(self: *@This()) c.heteris_runtime_provider_api_v1 {
        return .{ .abi_version = 1, .struct_size = @sizeOf(c.heteris_runtime_provider_api_v1), .authority = self, .provider_retain = retain, .provider_release = release, .cpu_query = query, .same_memory_domain = same, .scratch_allocate = allocate, .scratch_retain = scratchRetain, .scratch_release = scratchRelease, .cpu_bulk_sync = bulk };
    }
    fn descriptor(self: *@This(), api_ptr: *const c.heteris_runtime_provider_api_v1) c.heteris_runtime_provider_v1 {
        return .{ .abi_version = 1, .struct_size = @sizeOf(c.heteris_runtime_provider_v1), .api = api_ptr, .provider = self.token(1, c.HETERIS_RP_PROVIDER) };
    }
};
fn close(ctx: *common.Context, report: *results.CallReport) !void {
    _ = ctx.requestClose();
    try std.testing.expectEqual(common.CloseResult.closed, ctx.tryProgressClose(report));
}

test "integrated copies descriptor and API, closes only its retained provider outside gate" {
    var fake: Fake = .{ .allocator = std.testing.allocator };
    var api = fake.api();
    var desc = fake.descriptor(&api);
    var report: results.CallReport = .{};
    var ctx = try common.Context.initIntegrated(&desc, .{}, &report);
    fake.context = &ctx;
    api.provider_release = null;
    desc.provider.serial = 999;
    try std.testing.expectEqual(@as(usize, 2), fake.references);
    var operation = try ctx.begin();
    var pin = try operation.pinResource();
    _ = ctx.requestClose();
    try std.testing.expectEqual(common.CloseResult.busy, ctx.tryProgressClose(&report));
    operation.end();
    try std.testing.expectEqual(common.CloseResult.waiting_resources, ctx.tryProgressClose(&report));
    pin.releaseAfterResourceFree();
    try close(&ctx, &report);
    try std.testing.expectEqual(@as(usize, 1), fake.references);
    try std.testing.expectEqual(common.CloseResult.closed, ctx.tryProgressClose(&report));
}
test "query and caps rejection retain rollback preserves raw original error" {
    var fake: Fake = .{ .allocator = std.testing.allocator, .query_status = -997 };
    const api = fake.api();
    const desc = fake.descriptor(&api);
    var report: results.CallReport = .{};
    try std.testing.expectError(error.ProviderRejected, common.Context.initIntegrated(&desc, .{}, &report));
    try std.testing.expectEqual(@as(i32, -997), report.provider.returned_status);
    try std.testing.expectEqual(@as(usize, 1), fake.references);
    fake.query_status = 0;
    fake.flags = 0;
    try std.testing.expectError(error.Unsupported, common.Context.initIntegrated(&desc, .{}, &report));
    try std.testing.expectEqual(@as(usize, 2), fake.releases);
}
test "normal provider release failure retains exact token and allows explicit retry" {
    var fake: Fake = .{ .allocator = std.testing.allocator, .release_status = -887 };
    const api = fake.api();
    const desc = fake.descriptor(&api);
    var report: results.CallReport = .{};
    var ctx = try common.Context.initIntegrated(&desc, .{}, &report);
    fake.context = &ctx;
    _ = ctx.requestClose();
    try std.testing.expectEqual(common.CloseResult.release_failed, ctx.tryProgressClose(&report));
    try std.testing.expectEqual(@as(i32, -887), report.provider.returned_status);
    try std.testing.expectEqual(@as(usize, 2), fake.references);
    fake.release_status = 0;
    try close(&ctx, &report);
    try std.testing.expectEqual(@as(usize, 1), fake.references);
}
const Task = struct {
    value: *usize,
    fn call(self: @This()) void {
        self.value.* += 1;
    }
};
test "local descriptor and retain failures never acquire provider ownership" {
    var fake: Fake = .{ .allocator = std.testing.allocator, .retain_status = -651 };
    var api = fake.api();
    var desc = fake.descriptor(&api);
    var report: results.CallReport = .{};
    api.struct_size = 0;
    try std.testing.expectError(error.InvalidProvider, common.Context.initIntegrated(&desc, .{}, &report));
    try std.testing.expectEqual(@as(usize, 0), fake.retains);
    api = fake.api();
    try std.testing.expectError(error.ProviderRejected, common.Context.initIntegrated(&desc, .{}, &report));
    try std.testing.expectEqual(results.ResourceStage.retain, report.provider.stage);
    try std.testing.expectEqual(@as(i32, -651), report.provider.returned_status);
    try std.testing.expectEqual(@as(usize, 1), fake.references);
    try std.testing.expectEqual(@as(usize, 0), fake.releases);
    fake.retain_status = 0;
    try std.testing.expectError(error.Unsupported, common.Context.initIntegrated(&desc, .{ .max_parallel_tasks = 3 }, &report));
    try std.testing.expectEqual(@as(usize, 1), fake.references);
}
test "scratch extent and actual OOM preserve status and rollback their pin" {
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 0 });
    var fake: Fake = .{ .allocator = failing.allocator() };
    const api = fake.api();
    const desc = fake.descriptor(&api);
    var report: results.CallReport = .{};
    var ctx = try common.Context.initIntegrated(&desc, .{}, &report);
    fake.context = &ctx;
    var operation = try ctx.begin();
    const call = resources.admittedResources(&operation);
    try std.testing.expectError(error.ExtentOverflow, call.acquireScratch(f64, std.math.maxInt(usize), &report));
    try std.testing.expectEqual(@as(usize, 0), fake.allocation_calls);
    try std.testing.expectError(error.OutOfMemory, call.acquireScratch(u8, 1, &report));
    try std.testing.expectEqual(@as(i32, c.HETERIS_OUT_OF_MEMORY), report.provider.returned_status);
    try std.testing.expectEqual(@as(usize, 0), ctx.snapshot().resource_pins);
    try std.testing.expectEqual(@as(usize, 1), ctx.snapshot().active_operations);
    operation.end();
    try close(&ctx, &report);
}
test "zero-length aligned provider scratch retains pin after admitted operation ends" {
    const Aligned = struct { value: u8 align(64) };
    var fake: Fake = .{ .allocator = std.testing.allocator };
    const api = fake.api();
    const desc = fake.descriptor(&api);
    var report: results.CallReport = .{};
    var ctx = try common.Context.initIntegrated(&desc, .{}, &report);
    fake.context = &ctx;
    var operation = try ctx.begin();
    _ = ctx.requestClose();
    var scratch = try resources.admittedResources(&operation).acquireScratch(Aligned, 0, &report);
    try std.testing.expectEqual(@as(usize, 0), scratch.values.len);
    try std.testing.expectEqual(@as(usize, 0), @intFromPtr(scratch.values.ptr) % 64);
    try std.testing.expectEqual(@as(u64, 63), fake.used);
    operation.end();
    try std.testing.expectEqual(common.CloseResult.waiting_resources, ctx.tryProgressClose(&report));
    scratch.release(&report);
    try std.testing.expectEqual(@as(u64, 0), fake.used);
    try close(&ctx, &report);
}
test "concurrent finalizers observe busy while sole release runs outside gate" {
    var fake: Fake = .{ .allocator = std.testing.allocator, .pause_release = true };
    const api = fake.api();
    const desc = fake.descriptor(&api);
    var report: results.CallReport = .{};
    var ctx = try common.Context.initIntegrated(&desc, .{}, &report);
    fake.context = &ctx;
    _ = ctx.requestClose();
    const Finalizer = struct {
        ctx: *common.Context,
        result: common.CloseResult = undefined,
        fn run(self: *@This()) void {
            var r: results.CallReport = .{};
            self.result = self.ctx.tryProgressClose(&r);
        }
    };
    var finalizer: Finalizer = .{ .ctx = &ctx };
    const thread = try std.Thread.spawn(.{}, Finalizer.run, .{&finalizer});
    var joined = false;
    defer {
        fake.allow_release.store(true, .release);
        if (!joined) thread.join();
    }
    const started = std.Io.Clock.awake.now(std.testing.io);
    while (!fake.release_entered.load(.acquire) and started.durationTo(std.Io.Clock.awake.now(std.testing.io)).nanoseconds < 5 * std.time.ns_per_s) std.Thread.yield() catch {};
    try std.testing.expect(fake.release_entered.load(.acquire));
    try std.testing.expectEqual(common.CloseResult.busy, ctx.tryProgressClose(&report));
    try std.testing.expectEqual(common.Phase.releasing, ctx.snapshot().phase);
    fake.allow_release.store(true, .release);
    thread.join();
    joined = true;
    try std.testing.expectEqual(common.CloseResult.closed, finalizer.result);
    try std.testing.expectEqual(@as(usize, 1), fake.releases);
}
test "typed dispatch provider scratch, unknown reject and partial errors never replay" {
    var fake: Fake = .{ .allocator = std.testing.allocator };
    const api = fake.api();
    const desc = fake.descriptor(&api);
    var report: results.CallReport = .{};
    var ctx = try common.Context.initIntegrated(&desc, .{}, &report);
    fake.context = &ctx;
    var operation = try ctx.begin();
    const call = resources.admittedResources(&operation);
    var values = [_]usize{ 0, 0 };
    const tasks = [_]Task{ .{ .value = &values[0] }, .{ .value = &values[1] } };
    try call.runTypedChecked(Task, Task.call, &tasks, &report);
    try std.testing.expectEqualSlices(usize, &.{ 1, 1 }, &values);
    try std.testing.expectEqual(@as(u64, 0), fake.used);
    fake.behavior = .reject;
    report = .{};
    try std.testing.expectError(error.ProviderRejected, call.runTypedChecked(Task, Task.call, &tasks, &report));
    try std.testing.expectEqual(@as(i32, -778), report.provider.returned_status);
    try std.testing.expectEqual(results.OutputState.unchanged, report.output);
    try std.testing.expectEqualSlices(usize, &.{ 1, 1 }, &values);
    fake.behavior = .partial;
    report = .{};
    try std.testing.expectError(error.ProviderCallbackFailed, call.runTypedChecked(Task, Task.call, &tasks, &report));
    try std.testing.expectEqual(@as(i32, -1903), report.provider.callback_status);
    try std.testing.expectEqualSlices(usize, &.{ 2, 1 }, &values);
    fake.behavior = .malformed_range;
    report = .{};
    try std.testing.expectError(error.ProviderCallbackFailed, call.runTypedChecked(Task, Task.call, &tasks, &report));
    try std.testing.expectEqual(@as(i32, c.HETERIS_OUT_OF_BOUNDS), report.provider.callback_status);
    try std.testing.expectEqualSlices(usize, &.{ 2, 1 }, &values);
    fake.budget = 0;
    report = .{};
    try std.testing.expectError(error.ResourceBudgetExceeded, call.acquireScratch(u8, 1, &report));
    try std.testing.expectEqual(@as(usize, 0), ctx.snapshot().resource_pins);
    operation.end();
    try close(&ctx, &report);
}

pub fn failStopFixture(mode: []const u8) void {
    var fake: Fake = .{ .allocator = std.heap.c_allocator };
    const api = fake.api();
    const desc = fake.descriptor(&api);
    var report: results.CallReport = .{};
    if (std.mem.eql(u8, mode, "rollback")) {
        fake.query_status = -997;
        fake.release_status = -663;
        _ = common.Context.initIntegrated(&desc, .{}, &report) catch @panic("rollback incorrectly returned");
        @panic("rollback incorrectly succeeded");
    }
    var ctx = common.Context.initIntegrated(&desc, .{}, &report) catch @panic("fixture init");
    var operation = ctx.begin() catch @panic("fixture begin");
    if (std.mem.eql(u8, mode, "scratch")) {
        var scratch = resources.admittedResources(&operation).acquireScratch(u8, 1, &report) catch @panic("fixture scratch");
        fake.scratch_release_status = -664;
        scratch.release(&report);
    }
    fake.behavior = if (std.mem.eql(u8, mode, "counts")) .malformed_counts else .malformed_completion;
    var value: usize = 0;
    const tasks = [_]Task{.{ .value = &value }};
    resources.admittedResources(&operation).runTypedChecked(Task, Task.call, &tasks, &report) catch @panic("malformed completion incorrectly returned");
    @panic("fail-stop fixture incorrectly returned");
}
