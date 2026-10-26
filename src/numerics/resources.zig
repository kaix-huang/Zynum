// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Internal call-local scratch ownership, independent of numerical modules.
//! No default allocator, cache, pool, shared budget, or shutdown state lives here.

const std = @import("std");

/// An allocation owned for one synchronous call. The allocator's backing state
/// remains caller-owned and must outlive the lease. `values` is borrowed until
/// `deinit`; it is uninitialized on acquisition. Do not copy a live lease, resize
/// or free its slice separately, or retain a borrow after release. Deinitialize
/// exactly once; this type does not make allocator/thread safety guarantees.
///
/// Consumers retain their own admission/budget policy before acquisition.
/// Allocator.alloc supplies its normal checked element-byte multiplication and
/// OutOfMemory behavior, including a valid empty slice for a zero-sized request.
/// A later change to the caller's allocator variable cannot change who releases
/// this allocation. A future resource owner can supply its allocator here.
pub fn ScratchLease(comptime T: type) type {
    return struct {
        allocator: std.mem.Allocator,
        values: []T,

        pub fn acquire(allocator: std.mem.Allocator, count: usize) std.mem.Allocator.Error!@This() {
            return .{ .allocator = allocator, .values = try allocator.alloc(T, count) };
        }

        pub fn deinit(self: *@This()) void {
            self.allocator.free(self.values);
            self.* = undefined;
        }
    };
}
