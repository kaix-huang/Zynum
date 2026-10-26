// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Private executable root for isolated runtime lifecycle correctness checks.
//! Not imported or exported by any installed library.

const std = @import("std");
const runtime = @import("runtime.zig");
const pool = @import("core/execution/thread_pool.zig");

const Work = struct {
    caller_id: std.Thread.Id,
    visits: [2]std.atomic.Value(u32) = @splat(.init(0)),
    helper_observed: std.atomic.Value(bool) = .init(false),

    fn execute(raw: *const anyopaque, index: usize) void {
        const self: *@This() = @ptrCast(@alignCast(@constCast(raw)));
        if (index >= self.visits.len) @panic("invalid task index");
        _ = self.visits[index].fetchAdd(1, .monotonic);
        if (index == 1 and std.Thread.getCurrentId() != self.caller_id)
            self.helper_observed.store(true, .release);
    }
};

// C main owns process initialization; no std.process.Init/test runner Io exists.
export fn zynum_test_runtime_run(persistent: c_int) c_int {
    runtime.setMaxThreads(2);
    if (runtime.helperThreadCount(1) != 1) return 1;
    var work: Work = .{ .caller_id = std.Thread.getCurrentId() };
    const accepted = if (persistent != 0)
        pool.runLowLatency(Work.execute, &work, 2)
    else
        pool.run(Work.execute, &work, 2);
    if (!accepted) return 2;
    if (work.visits[0].load(.acquire) != 1 or work.visits[1].load(.acquire) != 1) return 3;
    if (!work.helper_observed.load(.acquire)) return 4;
    return 0;
}

export fn zynum_test_runtime_shutdown() void {
    pool.shutdown();
}

// Explicit C entry avoids std.start constructing an Io instance. The C driver
// installs/query-restores sentinels and controls all pool calls.
extern fn zynum_test_runtime_main(argc: c_int, argv: [*][*:0]u8) callconv(.c) c_int;
pub export fn main(argc: c_int, argv: [*][*:0]u8) callconv(.c) c_int {
    return zynum_test_runtime_main(argc, argv);
}

// Private, uninstalled fault entry: actual Io Group.Task allocation failures.
export fn zynum_test_runtime_fault(persistent: c_int, allow_one: c_int) c_int {
    const owner = @import("zynum-execution").owner;
    const fixture = @import("zynum-execution").test_support;
    fixture.setThreads(3);
    const limit: usize = if (allow_one != 0) 1 else 0;
    var allocations: fixture.Allocations = .{ .budget = limit };
    var executor = owner.Executor.init(allocations.allocator(), fixture.policy);
    defer executor.drainAndReset();
    var work: Work = .{ .caller_id = std.Thread.getCurrentId() };
    const batch: owner.Batch = .{ .task_fn = Work.execute, .tasks = &work, .count = 2 };
    const accepted = if (persistent != 0) executor.runLowLatency(batch) else executor.run(batch);
    if (!accepted) return 1;
    if (work.visits[0].load(.acquire) != 1 or work.visits[1].load(.acquire) != 1) return 2;
    if (work.helper_observed.load(.acquire) != (limit == 1)) return 3;
    executor.drainAndReset();
    const stats = allocations.stats();
    if (stats.accepted != limit or stats.freed != limit) return 4;
    if ((limit == 0 or persistent != 0) and stats.rejected == 0) return 5;
    return 0;
}
