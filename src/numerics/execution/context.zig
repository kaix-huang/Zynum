// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Internal synchronous operation lifetime. Keep Context at a stable address;
//! its external owner also keeps the allocator backing state and callback code
//! alive through close and every caller's return. This is not a public Context,
//! raw-pointer reclamation protocol, independent thread policy, or unload gate.
const std = @import("std");
const builtin = @import("builtin");
const owner = @import("owner.zig");

pub const Phase = enum { open, closing, draining, drained, closed };
pub const Lifetime = enum { immortal_default, internal_closable };
pub const AdmissionError = error{ Closing, CounterOverflow };
pub const CloseResult = enum { immortal, busy, waiting_resources, closed };
pub const RequestResult = enum { immortal, pending, closed };
pub const Snapshot = struct { phase: Phase, active_operations: usize, resource_pins: usize };
pub const TestEvent = enum { admitted, finishing };
const TestState = if (builtin.is_test) struct {
    limit: usize = std.math.maxInt(usize),
    observer: ?*const fn (*Context, TestEvent) void = null,
} else void;

pub const Context = struct {
    allocator: std.mem.Allocator,
    executor: owner.Executor,
    lifetime: Lifetime,
    gate: std.atomic.Value(bool) = .init(false),
    phase: Phase = .open,
    active_operations: usize = 0,
    resource_pins: usize = 0,
    test_state: TestState = if (builtin.is_test) .{} else {},

    /// No allocation or Io initialization. Do not move after first use.
    /// Policy remains the supplied static callbacks; no per-instance policy
    /// state is invented. A closable executor must only be used under begin().
    pub fn init(allocator: std.mem.Allocator, policy: owner.Policy, lifetime: Lifetime) Context {
        return .{ .allocator = allocator, .executor = owner.Executor.init(allocator, policy), .lifetime = lifetime };
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
        if (self.lifetime == .immortal_default) return .immortal;
        if (self.phase == .open) self.phase = .closing;
        return if (self.phase == .closed) .closed else .pending;
    }
    /// Never waits for active callers or outstanding resource pins. The one
    /// finalizer can synchronously wait for worker cleanup/Io teardown. It never
    /// holds gate during callbacks. Worker-owned pins must be released by drain
    /// before waiting on the remaining caller-owned resources.
    pub fn tryProgressClose(self: *Context) CloseResult {
        self.lock();
        if (self.lifetime == .immortal_default) {
            self.unlock();
            return .immortal;
        }
        switch (self.phase) {
            .open, .draining => {
                self.unlock();
                return .busy;
            },
            .closed => {
                self.unlock();
                return .closed;
            },
            .closing => {
                if (self.active_operations != 0) {
                    self.unlock();
                    return .busy;
                }
                self.phase = .draining;
                self.unlock();
                self.executor.drainAndReset();
                self.lock();
                self.phase = .drained;
            },
            .drained => {},
        }
        defer self.unlock();
        if (self.resource_pins != 0) return .waiting_resources;
        self.phase = .closed;
        return .closed;
    }
    /// Test configuration is quiescent, never a production callback/API.
    pub const testing = if (builtin.is_test) struct {
        pub fn configure(self: *Context, count_limit: usize, observer: ?*const fn (*Context, TestEvent) void) void {
            self.lock();
            defer self.unlock();
            if (self.phase != .open or self.active_operations != 0 or self.resource_pins != 0)
                @panic("Context testing configuration requires quiescence");
            self.test_state = .{ .limit = count_limit, .observer = observer };
        }
    } else struct {};
};

/// Internal affine value: no copying, async escape or use after end. The library
/// acquires it at an entrypoint; foreign callers do not have to manage a token.
pub const Operation = struct {
    context: ?*Context,
    pub fn borrowedContext(self: *const Operation) *Context {
        return self.context orelse @panic("operation already ended");
    }
    pub fn allocator(self: *const Operation) std.mem.Allocator {
        return self.borrowedContext().allocator;
    }
    pub fn pinResource(self: *const Operation) error{CounterOverflow}!ResourcePin {
        const ctx = self.borrowedContext();
        ctx.lock();
        defer ctx.unlock();
        if (ctx.active_operations == 0) @panic("resource pin requires a live operation");
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

/// One pin per owned resource/control, not a Context pointer-reclamation scheme.
/// Free the resource with its captured allocator before releasing this affine
/// value. Releasing does not implicitly drain or call arbitrary destructors.
pub const ResourcePin = struct {
    context: ?*Context,
    pub fn releaseAfterResourceFree(self: *ResourcePin) void {
        const ctx = self.context orelse @panic("resource pin already released");
        ctx.lock();
        defer ctx.unlock();
        if (ctx.resource_pins == 0) @panic("resource pin count underflow");
        ctx.resource_pins -= 1;
        self.context = null;
    }
};

/// Internal managed scratch. This affine lease owns an allocation plus one pin
/// on the original operation's Context. Do not copy it, mutate/free its inner
/// lease separately, or use a values borrow after deinit. The external Context
/// owner keeps its address and allocator/code backing state valid through close
/// and every API stack's return; the pin is not arbitrary pointer reclamation.
///
/// Acquire requires a live synchronous Operation and remains allowed after a
/// close request for that admitted operation. A successful lease may outlive the
/// operation (for example in a worker-owned cache), but must be deinitialized
/// exactly once before its Context can close. No new operation is needed for
/// deinit. Allocator callbacks execute outside the Context gate.
///
/// Zero-length and zero-sized-element requests retain ScratchLease's allocator
/// semantics and still own a pin until deinit. No budget, default, pool or
/// automatic finalizer is introduced; allocator/thread safety stays external.
pub fn ContextScratch(comptime T: type) type {
    const resources = @import("zynum-numerics-resources");
    return struct {
        lease: resources.ScratchLease(T),
        pin: ResourcePin,

        pub fn acquire(operation: *const Operation, count: usize) (std.mem.Allocator.Error || error{CounterOverflow})!@This() {
            var pin = try operation.pinResource();
            errdefer pin.releaseAfterResourceFree();
            const lease = try resources.ScratchLease(T).acquire(operation.allocator(), count);
            return .{ .lease = lease, .pin = pin };
        }

        pub fn deinit(self: *@This()) void {
            self.lease.deinit();
            self.pin.releaseAfterResourceFree();
            self.* = undefined;
        }
    };
}
