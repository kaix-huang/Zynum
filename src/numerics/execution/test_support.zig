// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Private fixtures only: imported by tests and the uninstalled lifecycle probe.
const std = @import("std");
const owner = @import("owner.zig");

var thread_limit: std.atomic.Value(usize) = .init(4);
pub fn setThreads(n: usize) void {
    thread_limit.store(n, .release);
}
fn maxThreads() usize {
    return thread_limit.load(.acquire);
}
fn helperCount(limit: usize) usize {
    return @min(limit, maxThreads() -| 1);
}
fn configureThread(_: ?usize) void {}
pub const policy: owner.Policy = .{
    .max_threads = maxThreads,
    .helper_count = helperCount,
    .configure_thread = configureThread,
    .worker_stack_size = 2 * 1024 * 1024,
};

pub fn wait(flag: *const std.atomic.Value(bool)) void {
    while (!flag.load(.acquire)) std.Thread.yield() catch {};
}

/// Lock protects all bookkeeping; backing allocator is c_allocator. Each actual
/// Group.Task request reaches this vtable. No scheduler result is synthesized.
pub const Allocations = struct {
    lock: std.atomic.Value(bool) = .init(false),
    budget: usize,
    attempts: usize = 0,
    accepted: usize = 0,
    rejected: usize = 0,
    freed: usize = 0,
    entries: [256]?[*]u8 = @splat(null),
    lengths: [256]usize = @splat(0),
    alignments: [256]std.mem.Alignment = undefined,

    pub const Stats = struct { attempts: usize, accepted: usize, rejected: usize, freed: usize };
    fn acquire(self: *Allocations) void {
        while (self.lock.cmpxchgStrong(false, true, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
    }
    fn release(self: *Allocations) void {
        self.lock.store(false, .release);
    }
    pub fn stats(self: *Allocations) Stats {
        self.acquire();
        defer self.release();
        return .{ .attempts = self.attempts, .accepted = self.accepted, .rejected = self.rejected, .freed = self.freed };
    }
    pub fn allocator(self: *Allocations) std.mem.Allocator {
        return .{ .ptr = self, .vtable = &.{ .alloc = alloc, .resize = resize, .remap = remap, .free = free } };
    }
    fn cast(p: *anyopaque) *Allocations {
        return @ptrCast(@alignCast(p));
    }
    fn find(self: *Allocations, memory: []u8, alignment: std.mem.Alignment) usize {
        for (self.entries, 0..) |p, i| if (p == memory.ptr) {
            std.debug.assert(self.lengths[i] == memory.len and self.alignments[i] == alignment);
            return i;
        };
        @panic("free/resize through a different allocator or unknown allocation");
    }
    fn alloc(p: *anyopaque, n: usize, a: std.mem.Alignment, ret: usize) ?[*]u8 {
        const self = cast(p);
        self.acquire();
        defer self.release();
        self.attempts += 1;
        if (self.accepted >= self.budget) {
            self.rejected += 1;
            return null;
        }
        const ptr = std.heap.c_allocator.rawAlloc(n, a, ret) orelse {
            self.rejected += 1;
            return null;
        };
        for (&self.entries, 0..) |*slot, i| if (slot.* == null) {
            slot.* = ptr;
            self.lengths[i] = n;
            self.alignments[i] = a;
            self.accepted += 1;
            return ptr;
        };
        @panic("private allocation record capacity exceeded");
    }
    fn resize(p: *anyopaque, memory: []u8, a: std.mem.Alignment, n: usize, ret: usize) bool {
        const self = cast(p);
        self.acquire();
        defer self.release();
        const i = self.find(memory, a);
        if (!std.heap.c_allocator.rawResize(memory, a, n, ret)) return false;
        self.lengths[i] = n;
        return true;
    }
    fn remap(p: *anyopaque, memory: []u8, a: std.mem.Alignment, n: usize, ret: usize) ?[*]u8 {
        const self = cast(p);
        self.acquire();
        defer self.release();
        const i = self.find(memory, a);
        const ptr = std.heap.c_allocator.rawRemap(memory, a, n, ret) orelse return null;
        self.entries[i] = ptr;
        self.lengths[i] = n;
        return ptr;
    }
    fn free(p: *anyopaque, memory: []u8, a: std.mem.Alignment, ret: usize) void {
        const self = cast(p);
        self.acquire();
        defer self.release();
        const i = self.find(memory, a);
        self.entries[i] = null;
        self.freed += 1;
        std.heap.c_allocator.rawFree(memory, a, ret);
    }
};
