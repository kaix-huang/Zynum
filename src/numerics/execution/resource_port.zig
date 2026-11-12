// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//! Private Zig capabilities. This is not a public C provider ABI or a resource owner.
//! The caller keeps owner, callback code and backing storage at stable addresses
//! until Context close and every admitted call have returned. No implicit resources.
const std = @import("std");
const results = @import("results.zig");
const contracts = @import("contracts.zig");

/// Bounded type erasure for an affine release obligation, not an address-derived
/// handle or ledger. An adapter stores its actual private value without decoding
/// another implementation's representation. No live obligation may be copied.
pub const Ownership = struct {
    bytes: [64]u8 align(16),
    pub fn init(comptime T: type, payload: T) Ownership {
        comptime {
            if (@sizeOf(T) > 64 or @alignOf(T) > 16) @compileError("release obligation exceeds private erasure capacity");
        }
        var storage: Ownership = undefined;
        @as(*T, @ptrCast(@alignCast(&storage.bytes))).* = payload;
        return storage;
    }
    pub fn value(self: *const Ownership, comptime T: type) *const T {
        comptime {
            if (@sizeOf(T) > 64 or @alignOf(T) > 16) @compileError("release obligation exceeds private erasure capacity");
        }
        return @ptrCast(@alignCast(&self.bytes));
    }
};
pub const ByteLease = struct {
    values: []u8,
    alignment: usize,
    ownership: Ownership,
    release_fn: *const fn (*const Ownership, *results.CallReport) void,
    live: bool = true,
    /// Callback returns only after release succeeds. An unprovable release must
    /// fail-stop while the exact obligation and Context pin remain live.
    pub fn release(self: *ByteLease, report: *results.CallReport) void {
        if (!self.live) @panic("scratch already released");
        self.release_fn(&self.ownership, report);
        self.live = false;
        self.values = undefined;
    }
};
/// The call-owned control block survives until run_sync proves rejected-without-
/// access or fully quiescent return. Async execution/cancellation is unsupported.
pub const TaskBatch = struct { task_fn: contracts.TaskFn, tasks: *const anyopaque, count: usize };
pub const VTable = struct {
    acquire_bytes: *const fn (*anyopaque, usize, usize, *results.CallReport) results.ExecutionError!ByteLease,
    run_sync: *const fn (*anyopaque, *const TaskBatch, *results.CallReport) results.ExecutionError!void,
    /// Called outside the Context gate only after operations and pins reach zero.
    /// False preserves the original provider obligation in release_failed state;
    /// the adapter alone interprets and reports its underlying release result.
    release_provider: *const fn (*anyopaque, *results.CallReport) bool,
};
pub const ResourcePort = struct {
    owner: *anyopaque,
    vtable: *const VTable,
    parallel_limit: usize,
    pub fn acquireBytes(self: ResourcePort, bytes: usize, alignment: usize, report: *results.CallReport) results.ExecutionError!ByteLease {
        return self.vtable.acquire_bytes(self.owner, bytes, alignment, report);
    }
    pub fn runSync(self: ResourcePort, batch: *const TaskBatch, report: *results.CallReport) results.ExecutionError!void {
        return self.vtable.run_sync(self.owner, batch, report);
    }
    pub fn releaseProvider(self: ResourcePort, report: *results.CallReport) bool {
        return self.vtable.release_provider(self.owner, report);
    }
};
