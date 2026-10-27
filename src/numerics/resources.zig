// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Internal call-local scratch ownership, independent of numerical modules.
//! No default allocator, cache, pool, shared budget, or shutdown state lives here.

const std = @import("std");
const c = @import("heteris-provider-c");

pub const ProviderReleaseFn = *const fn (c.heteris_rp_token) callconv(.c) c.heteris_status;

/// Affine primitive only: no Context import or access/quiescence permission.
/// Execution validates every descriptor field before calling initValidated.
/// A failed raw release preserves the exact token for its owner's fail-stop
/// policy; the primitive never reconstructs ownership from a data pointer.
pub fn ProviderScratchLease(comptime T: type) type {
    if (@sizeOf(T) == 0) @compileError("provider scratch requires nonzero-sized elements");
    return struct {
        token: c.heteris_rp_token,
        release_fn: ProviderReleaseFn,
        values: []T,
        live: bool = true,

        pub fn initValidated(desc: c.heteris_rp_scratch_v1, release_fn: ProviderReleaseFn, count: usize) @This() {
            const pointer: [*]T = @ptrCast(@alignCast(desc.data.?));
            return .{ .token = desc.lease, .release_fn = release_fn, .values = pointer[0..count] };
        }
        pub fn releaseRaw(self: *@This()) i32 {
            if (!self.live) @panic("provider scratch already released");
            const status = self.release_fn(self.token);
            if (status == c.HETERIS_OK) {
                self.live = false;
                self.values = undefined;
            }
            return status;
        }
    };
}

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
