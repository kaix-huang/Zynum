// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");
const binding = @import("sparse-interface-owner");
const Core = @import("zynum-core");
const execution = @import("zynum-execution");
const resources = @import("zynum-numerics-resources");
const fixture = execution.test_support;
const ctx_types = execution.context;
const Context = ctx_types.Context;
const Handles = binding.Handles;
const Adapter = binding.Adapter;
const equal = std.testing.expectEqual;
const expect = std.testing.expect;

/// Actual allocation provenance; callbacks deliberately run after rawFree, when
/// reading the destroyed shell would already be invalid. Only other handles may
/// be reentered. Ordinary tests use the real leak-checking testing allocator.
pub const Spy = struct {
    const Slot = struct { ptr: [*]u8, len: usize, alignment: std.mem.Alignment, id: usize };
    ctx: *Context = undefined,
    slots: [32]?Slot = @splat(null),
    attempts: usize = 0,
    accepted: usize = 0,
    freed: usize = 0,
    fail_at: ?usize = null,
    free_order: [32]usize = undefined,
    on_alloc: ?*const fn (*Spy, usize) void = null,
    on_free: ?*const fn (*Spy, usize) void = null,
    other: ?Handles.Handle = null,
    callback_count: usize = 0,
    pause_id: ?usize = null,
    entered: std.atomic.Value(bool) = .init(false),
    resume_free: std.atomic.Value(bool) = .init(false),
    returned: std.atomic.Value(bool) = .init(false),

    pub fn allocator(self: *Spy) std.mem.Allocator {
        return .{ .ptr = self, .vtable = &.{ .alloc = alloc, .free = free, .resize = std.mem.Allocator.noResize, .remap = std.mem.Allocator.noRemap } };
    }
    fn pinPresent(self: *Spy) void {
        if (self.ctx.snapshot().resource_pins == 0) @panic("free/alloc lost the common control pin");
    }
    fn alloc(raw: *anyopaque, len: usize, alignment: std.mem.Alignment, ret: usize) ?[*]u8 {
        const self: *Spy = @ptrCast(@alignCast(raw));
        self.pinPresent();
        const attempt = self.attempts;
        self.attempts += 1;
        if (self.on_alloc) |callback| callback(self, attempt);
        if (self.fail_at == attempt) return null;
        const ptr = std.testing.allocator.rawAlloc(len, alignment, ret) orelse return null;
        for (&self.slots) |*slot| if (slot.* == null) {
            slot.* = .{ .ptr = ptr, .len = len, .alignment = alignment, .id = self.accepted };
            self.accepted += 1;
            return ptr;
        };
        @panic("fixture allocation record capacity");
    }
    fn free(raw: *anyopaque, memory: []u8, alignment: std.mem.Alignment, ret: usize) void {
        const self: *Spy = @ptrCast(@alignCast(raw));
        self.pinPresent();
        for (&self.slots) |*slot| if (slot.*) |entry| {
            if (entry.ptr == memory.ptr) {
                if (entry.len != memory.len or entry.alignment != alignment or
                    @intFromPtr(memory.ptr) % alignment.toByteUnits() != 0)
                    @panic("free length/alignment/allocator provenance mismatch");
                slot.* = null;
                self.free_order[self.freed] = entry.id;
                self.freed += 1;
                std.testing.allocator.rawFree(memory, alignment, ret);
                if (self.on_free) |callback| callback(self, entry.id);
                if (self.pause_id == entry.id) {
                    if (self.ctx.tryProgressClose() != .waiting_resources) @panic("closed before allocator free returned");
                    self.entered.store(true, .release);
                    fixture.wait(&self.resume_free);
                    self.pinPresent();
                    self.returned.store(true, .release);
                }
                return;
            }
        };
        @panic("caller-owned array freed, double free or wrong allocator");
    }
    pub fn balanced(self: *Spy) !void {
        try equal(self.accepted, self.freed);
        for (self.slots) |slot| try expect(slot == null);
    }
};

pub fn close(context: *Context) !void {
    _ = context.requestClose();
    try equal(ctx_types.CloseResult.closed, context.tryProgressClose());
}
fn releaseHandle(handle: Handles.Handle) void {
    Adapter.destroy(handle) catch @panic("valid handle destruction failed");
}
fn emptyInput() Adapter.Input(f64, i32) {
    return .{ .base = 0, .rows = 0, .columns = 0, .axis = .row, .starts = &.{}, .ends = &.{}, .ordinals = &.{}, .values = &.{} };
}

test "typed Sparse shell and all eight payload controls share actual Core Context Resources and scalar identity" {
    const sparse = @import("zynum-sparse");
    comptime {
        if (binding.implementation != Core or binding.Provider.OwnerIdentity != Core or Handles.OwnerIdentity != Core or
            binding.Provider.Execution != execution or binding.Provider.Resources != resources or
            Handles.Operation != execution.context.Operation or Handles.Context != execution.context.Context)
            @compileError("interface binding introduced another implementation/resource owner");
        if (binding.Provider.Scalar.ComplexF32 != Core.ComplexF32 or binding.Provider.Scalar.ComplexF64 != sparse.ComplexF64)
            @compileError("interface complex identity differs from the public Core facade");
        for (.{ f32, f64, Core.ComplexF32, Core.ComplexF64 }) |T| {
            for (.{ i32, i64 }) |I| {
                if (Handles.Control(T, I) != Core.internal.numerics.SparseObject.ManagedMatrixControl(execution.context, T, I, I))
                    @compileError("payload control is not the actual Core managed control");
            }
        }
    }
    try expect(binding.Provider.Execution.getContext() == execution.getContext());
    try expect(binding.Provider.Execution.get() == &execution.getContext().executor);
    fixture.setThreads(1);
    var spy: Spy = .{};
    var context = Context.init(spy.allocator(), fixture.policy, .internal_closable);
    spy.ctx = &context;
    var h: ?Handles.Handle = null;
    try Adapter.createBorrowed(f64, i32, &context, emptyInput(), &h);
    const control = try Handles.testing.retainControl(f64, i32, h.?);
    try equal(@as(usize, 2), control.references);
    try equal(@as(usize, 1), context.snapshot().resource_pins);
    try equal(@as(usize, 2), spy.accepted);
    try Adapter.destroy(h);
    try equal(@as(usize, 1), control.references);
    try equal(@as(usize, 1), context.snapshot().resource_pins);
    try equal(@as(usize, 1), spy.freed); // Shell only.
    control.release();
    try std.testing.expectEqualSlices(usize, &.{ 1, 0 }, spy.free_order[0..spy.freed]);
    try spy.balanced();
    try close(&context);
}

test "typed Sparse control and shell allocation failures roll back every allocation before unpin" {
    fixture.setThreads(1);
    for (0..2) |fail_at| {
        var spy: Spy = .{ .fail_at = fail_at };
        var context = Context.init(spy.allocator(), fixture.policy, .internal_closable);
        spy.ctx = &context;
        var output: ?Handles.Handle = null;
        try std.testing.expectError(error.OutOfMemory, Adapter.createBorrowed(f64, i32, &context, emptyInput(), &output));
        try expect(output == null);
        try equal(fail_at + 1, spy.attempts);
        try equal(@as(usize, 0), context.snapshot().resource_pins);
        try equal(@as(usize, 0), context.snapshot().active_operations);
        try spy.balanced();
        try close(&context);
    }
}

test "typed Sparse admitted create survives allocator close request and destruction needs no admission" {
    const Callback = struct {
        fn call(spy: *Spy, _: usize) void {
            spy.callback_count += 1;
            _ = spy.ctx.requestClose();
            if (spy.ctx.tryProgressClose() != .busy) @panic("admitted create did not cover allocator callback");
        }
    };
    fixture.setThreads(1);
    var spy: Spy = .{ .on_alloc = Callback.call };
    var context = Context.init(spy.allocator(), fixture.policy, .internal_closable);
    spy.ctx = &context;
    var output: ?Handles.Handle = null;
    try Adapter.createBorrowed(f64, i32, &context, emptyInput(), &output);
    try equal(@as(usize, 2), spy.callback_count);
    try equal(ctx_types.CloseResult.waiting_resources, context.tryProgressClose());
    var exported: Adapter.Export(f64, i32) = undefined;
    try std.testing.expectError(error.Closing, Adapter.exportBorrowed(f64, i32, output, .row, &exported));
    try Adapter.destroy(output);
    try spy.balanced();
    try close(&context);
}

test "typed Sparse pin and admission overflow stay distinct from allocator OOM" {
    fixture.setThreads(1);
    var spy: Spy = .{ .fail_at = 0 };
    var context = Context.init(spy.allocator(), fixture.policy, .internal_closable);
    spy.ctx = &context;
    Context.testing.configure(&context, 1, null);
    var operation = try context.begin();
    var pin = try operation.pinResource();
    var output: ?Handles.Handle = null;
    var malformed = emptyInput();
    malformed.base = 77;
    // Admission wins before even scalar metadata/null-output checks.
    try std.testing.expectError(error.CounterOverflow, Adapter.createBorrowed(f64, i32, &context, malformed, null));
    operation.end();
    try std.testing.expectError(error.InvalidBase, Adapter.createBorrowed(f64, i32, &context, malformed, &output));
    try std.testing.expectError(error.CounterOverflow, Adapter.createBorrowed(f64, i32, &context, emptyInput(), &output));
    try equal(@as(usize, 0), spy.attempts);
    pin.releaseAfterResourceFree();
    try std.testing.expectError(error.OutOfMemory, Adapter.createBorrowed(f64, i32, &context, emptyInput(), &output));
    try equal(@as(usize, 1), spy.attempts);
    try equal(@as(usize, 0), context.snapshot().resource_pins);
    try spy.balanced();
    try close(&context);
}

test "typed Sparse borrowed destruction ignores invalidated indices and never frees caller arrays" {
    fixture.setThreads(1);
    var spy: Spy = .{};
    var context = Context.init(spy.allocator(), fixture.policy, .internal_closable);
    spy.ctx = &context;
    var starts = [_]i64{0};
    var ends = [_]i64{1};
    var indices = [_]i64{0};
    var values = [_]f64{std.math.nan(f64)};
    const before: [@sizeOf(@TypeOf(values))]u8 = std.mem.asBytes(&values).*;
    var h: ?Handles.Handle = null;
    try Adapter.createBorrowed(f64, i64, &context, .{ .base = 0, .rows = 1, .columns = 1, .axis = .row, .starts = &starts, .ends = &ends, .ordinals = &indices, .values = &values }, &h);
    indices[0] = -5;
    ends[0] = -10;
    _ = context.requestClose();
    try Adapter.destroy(h);
    try equal(@as(usize, 2), spy.freed);
    try spy.balanced();
    try std.testing.expectEqualSlices(u8, &before, std.mem.asBytes(&values));
    try equal(@as(i64, -5), indices[0]);
    try equal(@as(i64, -10), ends[0]);
    try close(&context);
}

test "typed Sparse shell and control raw frees each keep close waiting until allocator returns" {
    fixture.setThreads(1);
    for ([_]usize{ 1, 0 }) |pause_id| {
        var spy: Spy = .{ .pause_id = pause_id };
        var context = Context.init(spy.allocator(), fixture.policy, .internal_closable);
        spy.ctx = &context;
        var h: ?Handles.Handle = null;
        try Adapter.createBorrowed(f64, i32, &context, emptyInput(), &h);
        _ = context.requestClose();
        const thread = try std.Thread.spawn(.{}, releaseHandle, .{h.?});
        var joined = false;
        defer {
            spy.resume_free.store(true, .release);
            if (!joined) thread.join();
        }
        fixture.wait(&spy.entered);
        try equal(ctx_types.CloseResult.waiting_resources, context.tryProgressClose());
        try equal(@as(usize, 1), context.snapshot().resource_pins);
        try equal(@as(usize, 0), context.snapshot().active_operations);
        try expect(!spy.returned.load(.acquire));
        spy.resume_free.store(true, .release);
        thread.join();
        joined = true;
        try expect(spy.returned.load(.acquire));
        try std.testing.expectEqualSlices(usize, &.{ 1, 0 }, spy.free_order[0..spy.freed]);
        try spy.balanced();
        try close(&context);
    }
}

test "typed Sparse free callback reenters only another live handle before and after close" {
    const Callback = struct {
        fn call(spy: *Spy, id: usize) void {
            if (id != 3) return; // Second shell; no reentry into this dead shell.
            spy.callback_count += 1;
            var view: Adapter.Export(f64, i32) = undefined;
            Adapter.exportBorrowed(f64, i32, spy.other, .row, &view) catch @panic("other live handle export failed");
            var temporary: ?Handles.Handle = null;
            Adapter.createBorrowed(f64, i32, spy.ctx, emptyInput(), &temporary) catch @panic("other create while open failed");
            Adapter.destroy(temporary) catch @panic("temporary destroy failed");
            _ = spy.ctx.requestClose();
            if (Adapter.exportBorrowed(f64, i32, spy.other, .row, &view)) |_| @panic("closing export admitted") else |err| {
                if (err != error.Closing) @panic("wrong closing failure");
            }
            if (Adapter.createBorrowed(f64, i32, spy.ctx, emptyInput(), &temporary)) |_| @panic("closing create admitted") else |err| {
                if (err != error.Closing) @panic("wrong closing failure");
            }
            Adapter.destroy(spy.other) catch @panic("other handle cannot be destroyed while closing");
            spy.other = null;
            if (spy.ctx.tryProgressClose() != .waiting_resources) @panic("destroy callback lost outer control pin");
        }
    };
    fixture.setThreads(1);
    var spy: Spy = .{ .on_free = Callback.call };
    var context = Context.init(spy.allocator(), fixture.policy, .internal_closable);
    spy.ctx = &context;
    var first: ?Handles.Handle = null;
    var second: ?Handles.Handle = null;
    try Adapter.createBorrowed(f64, i32, &context, emptyInput(), &first);
    try Adapter.createBorrowed(f64, i32, &context, emptyInput(), &second);
    spy.other = first;
    try Adapter.destroy(second);
    try equal(@as(usize, 1), spy.callback_count);
    try equal(@as(usize, 6), spy.accepted);
    try spy.balanced();
    try close(&context);
}

test "typed Sparse same Context export prerequisite precedes common format and structural validation" {
    fixture.setThreads(1);
    var context = Context.init(std.testing.allocator, fixture.policy, .internal_closable);
    var other = Context.init(std.testing.allocator, fixture.policy, .internal_closable);
    var h: ?Handles.Handle = null;
    try Adapter.createBorrowed(f64, i32, &context, emptyInput(), &h);
    var wrong = try other.begin();
    try std.testing.expectError(error.ContextMismatch, Handles.exportView(f64, i32, h.?, &wrong, .column));
    wrong.end();
    try equal(@as(usize, 1), context.snapshot().resource_pins);
    try Adapter.destroy(h);
    try close(&context);
    try close(&other);
}

const CachedControl = Handles.Control(f64, i32);
const CacheEntry = struct { control: *CachedControl, context: *Context, cleaned: *std.atomic.Value(bool) };
threadlocal var cached: ?CacheEntry = null;
fn releaseCache() void {
    const entry = cached orelse return;
    if (entry.context.tryProgressClose() != .busy) @panic("drain callback under gate or outside draining");
    if (entry.control.references != 1) @panic("cache did not retain the last control reference");
    entry.control.release();
    cached = null;
    entry.cleaned.store(true, .release);
}
test "typed Sparse shell destruction leaves worker retained common control for real close drain" {
    fixture.setThreads(2);
    var allocations: fixture.Allocations = .{ .budget = 100 };
    var context = Context.init(allocations.allocator(), fixture.policy, .internal_closable);
    defer context.executor.drainAndReset();
    var h: ?Handles.Handle = null;
    try Adapter.createBorrowed(f64, i32, &context, emptyInput(), &h);
    var operation = try context.begin();
    const Work = struct {
        entry: CacheEntry,
        fn task(raw: *const anyopaque, index: usize) void {
            if (index == 0) return;
            if (cached != null) @panic("unexpected stale test worker cache");
            const self: *const @This() = @ptrCast(@alignCast(raw));
            cached = self.entry;
        }
    };
    var cleaned: std.atomic.Value(bool) = .init(false);
    var work: Work = .{ .entry = .{ .control = try Handles.testing.retainControl(f64, i32, h.?), .context = &context, .cleaned = &cleaned } };
    const domain: execution.owner.WorkerCacheDomain = .{ .release_current_worker = releaseCache };
    const ran = context.executor.runLowLatency(.{ .task_fn = Work.task, .tasks = &work, .count = 2, .cache_domain = &domain });
    if (!ran) work.entry.control.release();
    try Adapter.destroy(h);
    operation.end();
    try expect(ran);
    try equal(@as(usize, 1), context.snapshot().resource_pins);
    try expect(!cleaned.load(.acquire));
    try close(&context);
    try expect(cleaned.load(.acquire));
    try equal(allocations.stats().accepted, allocations.stats().freed);
}
