// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Private, non-generic configuration owned by the one LAPACKE binding Module.
//! Never import this file from another module root: use that binding instead.
//! No allocator, Context, Executor, standard C symbol or numerical policy lives
//! here. The two returning-event channels are an internal test contract, not
//! LAPACKE_xerbla/mkl_set_xerbla ABI implementations or a full-library registry.
//!
//! NaN initialization follows Reference's defined sequential policy: lazy first
//! get, default on, cached environment, explicit zero/nonzero setter priority.
//! The ASCII decimal-prefix parser avoids atoi overflow: any nonzero prefix is
//! enabled even outside C int range. That overflow rule is a Zynum extension,
//! not Reference or MKL equivalence. MKL lazy timing/parsing is not established.
//! The process environment must remain stable while its first read occurs.
//!
//! Configuration operations are synchronized; callback invocation is not under
//! the gate. Referenced user state still requires its caller's synchronization.
//! Replacement does not reclaim old userdata/code: callers keep them
//! alive until all invocations that captured them return. No async-signal,
//! fork-lock recovery, arbitrary pointer, cross-DSO or unload safety is promised.
const std = @import("std");
const builtin = @import("builtin");
const common = @import("lapacke_common.zig");

pub const Callback = struct {
    userdata: *anyopaque,
    function: *const fn (*anyopaque, common.ErrorEvent) void,
};
pub const Callbacks = struct {
    lapacke: ?Callback = null,
    fortran_returning_test: ?Callback = null,

    /// A null channel is the explicit returning/no-output typed test sink.
    /// This uses only this captured value, never the current global binding.
    pub fn dispatch(self: *const Callbacks, event: common.ErrorEvent) void {
        const selected = switch (event.kind) {
            .lapacke => self.lapacke,
            .fortran_returning_test => self.fortran_returning_test,
        };
        if (selected) |callback| callback.function(callback.userdata, event);
    }
};

const NanState = enum { uninitialized, enabled, disabled };
const TestCounters = if (builtin.is_test) struct {
    getter_calls: usize = 0,
    environment_reads: usize = 0,
    captures: usize = 0,
} else void;
var state: struct {
    gate: std.atomic.Value(bool) = .init(false),
    nan: NanState = .uninitialized,
    callbacks: Callbacks = .{},
    counters: TestCounters = if (builtin.is_test) .{} else {},
} = .{};

fn lock() void {
    while (state.gate.cmpxchgStrong(false, true, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
}
fn unlock() void {
    state.gate.store(false, .release);
}

fn parseEnvironmentValue(text: []const u8) bool {
    var i: usize = 0;
    while (i < text.len and std.ascii.isWhitespace(text[i])) : (i += 1) {}
    if (i < text.len and (text[i] == '+' or text[i] == '-')) i += 1;
    var nonzero = false;
    while (i < text.len and std.ascii.isDigit(text[i])) : (i += 1) {
        nonzero = nonzero or text[i] != '0';
    }
    return nonzero;
}

/// Internal C-int-shaped control; not an exported standard ABI function.
pub fn setNancheck(flag: c_int) void {
    lock();
    defer unlock();
    state.nan = if (flag == 0) .disabled else .enabled;
}
pub fn getNancheck() c_int {
    lock();
    defer unlock();
    if (comptime builtin.is_test) state.counters.getter_calls += 1;
    if (state.nan == .uninitialized) {
        if (comptime builtin.is_test) state.counters.environment_reads += 1;
        // libc supplies the environment access; no user callback or allocation
        // is used by this owner. Do not change environ concurrently with this.
        const raw = std.c.getenv("LAPACKE_NANCHECK");
        const enabled = if (raw) |value| parseEnvironmentValue(std.mem.span(value)) else true;
        state.nan = if (enabled) .enabled else .disabled;
    }
    return @intFromBool(state.nan == .enabled);
}

/// Replace both explicitly separate typed channels as one coherent record.
/// The returned old record is not a reclamation/quiescence token.
pub fn setCallbacks(next: Callbacks) Callbacks {
    lock();
    defer unlock();
    const previous = state.callbacks;
    state.callbacks = next;
    return previous;
}
/// Capture functions and userdata, not the NaN flag. Only an admitted binding
/// call should use this snapshot to execute an adapter; no callback runs here.
pub fn captureCallbacks() Callbacks {
    return captureCallbacksWithDefaults(.{});
}

/// Per-call profile defaults fill only unconfigured channels. They are never
/// written to the shared configuration. Capture once after operation admission;
/// callers retain default userdata until the synchronous operation returns.
pub fn captureCallbacksWithDefaults(defaults: Callbacks) Callbacks {
    lock();
    defer unlock();
    if (comptime builtin.is_test) state.counters.captures += 1;
    return .{
        .lapacke = state.callbacks.lapacke orelse defaults.lapacke,
        .fortran_returning_test = state.callbacks.fortran_returning_test orelse defaults.fortran_returning_test,
    };
}

/// Quiescent fixture reset only; not present in production and not a way to
/// invalidate in-flight snapshots. Canonical fixture calls are serialized;
/// each fixture must join its own threads before reset or userdata teardown.
pub const testing = if (builtin.is_test) struct {
    pub const Snapshot = struct {
        initialized: bool,
        getter_calls: usize,
        environment_reads: usize,
        captures: usize,
    };
    pub fn reset() void {
        lock();
        defer unlock();
        state.nan = .uninitialized;
        state.callbacks = .{};
        state.counters = .{};
    }
    pub fn snapshot() Snapshot {
        lock();
        defer unlock();
        return .{ .initialized = state.nan != .uninitialized, .getter_calls = state.counters.getter_calls, .environment_reads = state.counters.environment_reads, .captures = state.counters.captures };
    }
    pub fn identity() *const anyopaque {
        return &state;
    }
    pub const parseEnvironment = parseEnvironmentValue;
} else struct {};
