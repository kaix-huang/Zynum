// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//! Ownerless synchronous lifetime. External storage/code survives all callers.
const std = @import("std");
const builtin = @import("builtin");
const options = @import("zynum-build-options");
const ports = @import("resource_port.zig");
const legacy = @import("legacy_services.zig");
const results = @import("results.zig");
const resources = @import("zynum-numerics-resources");
pub const Phase = enum { open, closing, draining, drained, releasing, release_failed, closed };
pub const AdmissionError = error{ Closing, CounterOverflow };
pub const CloseResult = enum { immortal, busy, waiting_resources, release_failed, closed };
pub const RequestResult = enum { immortal, pending, closed };
pub const Snapshot = struct { phase: Phase, active_operations: usize, resource_pins: usize };
pub const TestEvent = enum { admitted, finishing };
const TestState = if (builtin.is_test) struct { limit: usize = std.math.maxInt(usize), observer: ?*const fn (*Context, TestEvent) void = null } else void;
const Binding = union(enum) { external: ports.ResourcePort, legacy: legacy.LegacyServices };
pub const Context = struct {
    binding: Binding,
    gate: std.atomic.Value(bool) = .init(false),
    phase: Phase = .open,
    active_operations: usize = 0,
    resource_pins: usize = 0,
    test_state: TestState = if (builtin.is_test) .{} else {},

    /// Borrows an explicitly selected capability. Owner/code stay stable through
    /// close; production adapters perform their own external ABI validation.
    pub fn initResources(port: ports.ResourcePort) Context {
        if (port.parallel_limit == 0 or port.parallel_limit > @import("contracts.zig").max_tasks) @panic("invalid private resource port parallel limit");
        return .{ .binding = .{ .external = port } };
    }
    pub fn initLegacyBorrowed(services: legacy.LegacyServices) Context {
        if (comptime !options.reference_compat) @compileError("legacy Context construction requires reference_compat");
        return .{ .binding = .{ .legacy = services } };
    }
    pub fn resourcePort(self: *const Context) ?ports.ResourcePort {
        return switch (self.binding) {
            .external => |port| port,
            .legacy => null,
        };
    }
    fn immortal(self: *const Context) bool {
        return switch (self.binding) {
            .external => false,
            .legacy => |s| s.lifetime == .immortal_default,
        };
    }
    fn lock(self: *Context) void {
        while (self.gate.cmpxchgStrong(false, true, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
    }
    fn unlock(self: *Context) void {
        self.gate.store(false, .release);
    }
    fn limit(self: *const Context) usize {
        return if (builtin.is_test) self.test_state.limit else std.math.maxInt(usize);
    }
    fn observe(self: *Context, event: TestEvent) void {
        if (comptime builtin.is_test) if (self.test_state.observer) |callback| callback(self, event);
    }
    pub fn snapshot(self: *Context) Snapshot {
        self.lock();
        defer self.unlock();
        return .{ .phase = self.phase, .active_operations = self.active_operations, .resource_pins = self.resource_pins };
    }
    pub fn begin(self: *Context) AdmissionError!Operation {
        self.lock();
        if (self.phase != .open) {
            self.unlock();
            return error.Closing;
        }
        if (self.active_operations == self.limit()) {
            self.unlock();
            return error.CounterOverflow;
        }
        self.active_operations += 1;
        self.unlock();
        self.observe(.admitted);
        return .{ .context = self };
    }
    pub fn requestClose(self: *Context) RequestResult {
        self.lock();
        defer self.unlock();
        if (self.immortal()) return .immortal;
        if (self.phase == .open) self.phase = .closing;
        return if (self.phase == .closed) .closed else .pending;
    }
    pub fn tryProgressClose(self: *Context, report: *results.CallReport) CloseResult {
        report.* = .{};
        self.lock();
        if (self.immortal()) {
            self.unlock();
            return .immortal;
        }
        switch (self.phase) {
            .open, .draining, .releasing => {
                self.unlock();
                return .busy;
            },
            .closed => {
                self.unlock();
                return .closed;
            },
            .closing, .release_failed => {
                if (self.active_operations != 0) {
                    self.unlock();
                    return .busy;
                }
                switch (self.binding) {
                    .legacy => |services| {
                        self.phase = .draining;
                        self.unlock();
                        services.vtable.drain_owned(services.userdata);
                        self.lock();
                        self.phase = .drained;
                    },
                    .external => |port| {
                        if (self.resource_pins != 0) {
                            self.unlock();
                            return .waiting_resources;
                        }
                        self.phase = .releasing;
                        self.unlock();
                        const released = port.releaseProvider(report);
                        self.lock();
                        self.phase = if (released) .closed else .release_failed;
                        self.unlock();
                        return if (released) .closed else .release_failed;
                    },
                }
            },
            .drained => {},
        }
        defer self.unlock();
        if (self.resource_pins != 0) return .waiting_resources;
        self.phase = .closed;
        return .closed;
    }
    pub const testing = if (builtin.is_test) struct {
        pub fn configure(self: *Context, count_limit: usize, observer: ?*const fn (*Context, TestEvent) void) void {
            self.lock();
            defer self.unlock();
            if (self.phase != .open or self.active_operations != 0 or self.resource_pins != 0) @panic("Context testing configuration requires quiescence");
            self.test_state = .{ .limit = count_limit, .observer = observer };
        }
    } else struct {};
};
pub const Operation = struct {
    context: ?*Context,
    pub fn borrowedContext(self: *const Operation) *Context {
        return self.context orelse @panic("operation already ended");
    }
    pub fn legacyServices(self: *const Operation) legacy.LegacyServices {
        if (comptime !options.reference_compat) @compileError("legacy resources require reference_compat");
        return switch (self.borrowedContext().binding) {
            .legacy => |s| s,
            .external => @panic("integrated operation used as reference resources"),
        };
    }
    pub fn legacyAllocator(self: *const Operation) std.mem.Allocator {
        const s = self.legacyServices();
        return s.vtable.allocator(s.userdata);
    }
    /// Historical reference Core spelling. Never usable by integrated code.
    pub fn allocator(self: *const Operation) std.mem.Allocator {
        if (comptime !options.reference_compat) @compileError("Operation.allocator requires reference_compat; use provider scratch");
        return self.legacyAllocator();
    }
    pub fn pinResource(self: *const Operation) error{CounterOverflow}!ResourcePin {
        const ctx = self.borrowedContext();
        ctx.lock();
        defer ctx.unlock();
        if (ctx.active_operations == 0) @panic("pin requires a live operation");
        if (ctx.resource_pins == ctx.limit()) return error.CounterOverflow;
        ctx.resource_pins += 1;
        return .{ .context = ctx };
    }
    pub fn end(self: *Operation) void {
        const ctx = self.borrowedContext();
        ctx.observe(.finishing);
        ctx.lock();
        defer ctx.unlock();
        if (ctx.active_operations == 0) @panic("operation count underflow");
        ctx.active_operations -= 1;
        self.context = null;
    }
};
pub const ResourcePin = struct {
    context: ?*Context,
    pub fn releaseAfterResourceFree(self: *ResourcePin) void {
        const ctx = self.context orelse @panic("pin already released");
        ctx.lock();
        defer ctx.unlock();
        if (ctx.resource_pins == 0) @panic("pin count underflow");
        ctx.resource_pins -= 1;
        self.context = null;
    }
};
pub fn OperationScratch(comptime T: type) type {
    return struct {
        lease: ports.ByteLease,
        pin: ResourcePin,
        values: []T,
        pub fn release(self: *@This(), report: *results.CallReport) void {
            self.lease.release(report);
            self.pin.releaseAfterResourceFree();
            self.* = undefined;
        }
    };
}
pub fn acquireOperationScratch(operation: *const Operation, comptime T: type, count: usize, report: *results.CallReport) results.ExecutionError!OperationScratch(T) {
    const port = operation.borrowedContext().resourcePort() orelse return error.Unsupported;
    var pin = try operation.pinResource();
    errdefer pin.releaseAfterResourceFree();
    if (@sizeOf(T) == 0) @compileError("resource scratch requires nonzero-sized elements");
    const bytes = std.math.mul(usize, count, @sizeOf(T)) catch return error.ExtentOverflow;
    _ = std.math.add(usize, bytes, @alignOf(T) - 1) catch return error.ExtentOverflow;
    const lease = try port.acquireBytes(bytes, @alignOf(T), report);
    // An adapter guarantees exact extent/alignment and a nonnull even empty
    // backing; never infer ownership from this borrowed data pointer.
    if (lease.values.len != bytes or lease.alignment != @alignOf(T) or @intFromPtr(lease.values.ptr) % @alignOf(T) != 0) @panic("invalid private scratch capability");
    _ = std.math.add(usize, @intFromPtr(lease.values.ptr), bytes) catch @panic("private scratch extent overflow");
    const pointer: [*]T = @ptrCast(@alignCast(lease.values.ptr));
    return .{ .lease = lease, .pin = pin, .values = pointer[0..count] };
}
/// Explicit reference-only scratch. Integrated calls cannot use an allocator.
pub fn ContextScratch(comptime T: type) type {
    return struct {
        lease: resources.ScratchLease(T),
        pin: ResourcePin,
        pub fn acquire(operation: *const Operation, count: usize) (std.mem.Allocator.Error || error{CounterOverflow})!@This() {
            if (comptime !options.reference_compat) @compileError("legacy ContextScratch requires reference_compat");
            var pin = try operation.pinResource();
            errdefer pin.releaseAfterResourceFree();
            return .{ .lease = try resources.ScratchLease(T).acquire(operation.legacyAllocator(), count), .pin = pin };
        }
        pub fn deinit(self: *@This()) void {
            self.lease.deinit();
            self.pin.releaseAfterResourceFree();
            self.* = undefined;
        }
    };
}
