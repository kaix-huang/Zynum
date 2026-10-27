// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Internal synchronous execution owner. Keep a live instance at a stable address.
//! This is restartable reset, not an operation-wide close or unload barrier.
const std = @import("std");

const contracts = @import("contracts.zig");
pub const max_tasks = contracts.max_tasks;
pub const TaskFn = contracts.TaskFn;
const persistent_spin_iters: u32 = 500_000;
const persistent_done_spin_iters: u32 = 100_000;
pub const Policy = contracts.Policy;
/// Static descriptor and code must outlive every worker. Release is worker-local.
pub const WorkerCacheDomain = contracts.WorkerCacheDomain;
pub const Batch = contracts.Batch;

pub const Executor = struct {
    allocator: std.mem.Allocator,
    policy: Policy,
    // Immutable after reference binding initialization, before any worker use.
    default_cache_domain: ?*const WorkerCacheDomain = null,
    persistent_cache_domain: ?*const WorkerCacheDomain = null,
    io_init_state: std.atomic.Value(u8) = .init(0),
    io_busy: std.atomic.Value(u8) = .init(0),
    io_threaded: std.Io.Threaded = undefined,
    persistent_threaded: std.Io.Threaded = undefined,

    // Production access is serialized by io_busy; private test setup is quiescent.
    // Io.Threaded saves process signal handlers at init and restores them at deinit,
    // so release these owners in reverse order without changing either worker protocol.
    first_initialized_io: FirstInitializedIo = .none,

    persistent_init_state: std.atomic.Value(u8) = .init(0),
    persistent_group: std.Io.Group = .init,
    persistent_worker_count: std.atomic.Value(u32) = .init(0),
    persistent_ready_count: std.atomic.Value(u32) = .init(0),
    persistent_exited_count: std.atomic.Value(u32) = .init(0),
    persistent_worker_generation: [max_tasks]std.atomic.Value(u32) = @splat(std.atomic.Value(u32).init(0)),
    persistent_worker_waiting: [max_tasks]std.atomic.Value(u8) = @splat(std.atomic.Value(u8).init(0)),
    persistent_active_helpers: std.atomic.Value(u32) = .init(0),
    persistent_first_helper: std.atomic.Value(u32) = .init(0),
    persistent_done_target: std.atomic.Value(u32) = .init(0),
    persistent_done_count: std.atomic.Value(u32) = .init(0),
    persistent_job_count: std.atomic.Value(u32) = .init(0),
    persistent_shutdown_requested: std.atomic.Value(u8) = .init(0),
    persistent_task_fn: TaskFn = undefined,
    persistent_tasks: *const anyopaque = undefined,

    const FirstInitializedIo = enum { none, ordinary, persistent };

    /// Does not initialize Io or borrow this address. Do not move after first use.
    /// Allocator must be thread-safe and its state must outlive drainAndReset.
    pub fn init(allocator: std.mem.Allocator, policy: Policy) Executor {
        return .{ .allocator = allocator, .policy = policy };
    }

    fn configuredHelperCount(self: *Executor) usize {
        return self.policy.helper_count(max_tasks - 1);
    }

    fn ensureIoThreaded(self: *Executor) bool {
        if (self.configuredHelperCount() == 0) return false;

        const state = self.io_init_state.load(.acquire);
        if (state == 2) return true;
        if (state == 3) return false;

        if (self.io_init_state.cmpxchgStrong(0, 1, .acq_rel, .acquire) == null) {
            const helper_count = self.configuredHelperCount();
            if (helper_count == 0) {
                self.io_init_state.store(3, .release);
                return false;
            }
            self.io_threaded = std.Io.Threaded.init(self.allocator, .{
                .stack_size = self.policy.worker_stack_size,
                .async_limit = .limited(helper_count),
                .concurrent_limit = .limited(helper_count),
            });
            if (self.first_initialized_io == .none) self.first_initialized_io = .ordinary;
            self.io_init_state.store(2, .release);
            return true;
        }

        while (true) {
            const current = self.io_init_state.load(.acquire);
            if (current == 2) return true;
            if (current == 3) return false;
            std.atomic.spinLoopHint();
        }
    }

    fn runIoTask(self: *Executor, cache_domain: ?*const WorkerCacheDomain, task_fn: TaskFn, tasks: *const anyopaque, index: usize) void {
        self.policy.configure_thread(index);
        // Io.Threaded owns its worker lifetimes; release this task's thread-local
        // kernel allocations before returning to a thread we cannot finalize.
        defer if (cache_domain) |domain| domain.release_current_worker();
        task_fn(tasks, index);
    }

    fn runPersistentWorker(self: *Executor, worker_id: usize) void {
        self.policy.configure_thread(worker_id + 1);
        const io = self.persistent_threaded.io();
        var last_domain: ?*const WorkerCacheDomain = null;
        defer {
            // A TLS pointer disappearing at thread exit does not free its backing
            // allocation. Dispatch cleanup on the owner thread before reporting exit.
            if (last_domain) |domain| domain.release_current_worker();
            const exited = self.persistent_exited_count.fetchAdd(1, .acq_rel) + 1;
            if (exited >= self.persistent_worker_count.load(.acquire)) {
                io.futexWake(u32, &self.persistent_exited_count.raw, 1);
            }
        }
        var seen = self.persistent_worker_generation[worker_id].load(.acquire);
        _ = self.persistent_ready_count.fetchAdd(1, .acq_rel);
        io.futexWake(u32, &self.persistent_ready_count.raw, 1);

        while (true) {
            var current = self.persistent_worker_generation[worker_id].load(.acquire);
            var spin_count: u32 = 0;
            while (current == seen and spin_count < persistent_spin_iters) : (spin_count += 1) {
                std.atomic.spinLoopHint();
                current = self.persistent_worker_generation[worker_id].load(.acquire);
            }
            if (current == seen) {
                if (comptime @import("builtin").target.os.tag == .macos) self.persistent_worker_waiting[worker_id].store(1, .release);
                current = self.persistent_worker_generation[worker_id].load(.acquire);
                while (current == seen) {
                    io.futexWaitUncancelable(u32, &self.persistent_worker_generation[worker_id].raw, seen);
                    current = self.persistent_worker_generation[worker_id].load(.acquire);
                }
                if (comptime @import("builtin").target.os.tag == .macos) self.persistent_worker_waiting[worker_id].store(0, .release);
            }
            seen = current;
            if (self.persistent_shutdown_requested.load(.acquire) != 0) break;

            const task_fn = self.persistent_task_fn;
            const tasks = self.persistent_tasks;
            const count = self.persistent_job_count.load(.acquire);
            const done_target = self.persistent_done_target.load(.acquire);
            const active_helpers = self.persistent_active_helpers.load(.acquire);
            const first_helper = self.persistent_first_helper.load(.acquire);
            if (worker_id < first_helper or worker_id >= first_helper + active_helpers) continue;
            // The generation acquire also publishes the immutable cache domain.
            // Inactive helpers retain their previous domain until work or shutdown.
            const next_domain = self.persistent_cache_domain;
            if (last_domain != next_domain) {
                if (last_domain) |domain| domain.release_current_worker();
                last_domain = next_domain;
            }
            const index = worker_id - first_helper + 1;
            if (index < count) task_fn(tasks, @intCast(index));

            const done = self.persistent_done_count.fetchAdd(1, .acq_rel) + 1;
            if (done >= done_target) io.futexWake(u32, &self.persistent_done_count.raw, 1);
        }
    }

    fn ensurePersistentWorkers(self: *Executor) bool {
        if (self.configuredHelperCount() == 0) return false;

        const state = self.persistent_init_state.load(.acquire);
        if (state == 2) return true;
        if (state == 3) return false;

        if (self.persistent_init_state.cmpxchgStrong(0, 1, .acq_rel, .acquire) == null) {
            const helper_count = self.configuredHelperCount();
            if (helper_count == 0) {
                self.persistent_init_state.store(3, .release);
                return false;
            }
            self.persistent_shutdown_requested.store(0, .release);
            self.persistent_exited_count.store(0, .release);
            self.persistent_threaded = std.Io.Threaded.init(self.allocator, .{
                .stack_size = self.policy.worker_stack_size,
                .async_limit = .limited(helper_count),
                .concurrent_limit = .limited(helper_count),
            });

            const io = self.persistent_threaded.io();
            self.persistent_ready_count.store(0, .release);
            var submitted: u32 = 0;
            for (0..helper_count) |worker_id| {
                self.persistent_group.concurrent(io, runPersistentWorker, .{ self, worker_id }) catch break;
                submitted += 1;
            }
            if (submitted == 0) {
                // The Io is already owned even though no task was accepted. Pair
                // init now; state 3 must not retain an object that shutdown skips.
                self.persistent_threaded.deinit();
                self.persistent_group = .init;
                self.persistent_init_state.store(3, .release);
                return false;
            }
            self.persistent_worker_count.store(submitted, .release);

            while (true) {
                const ready = self.persistent_ready_count.load(.acquire);
                if (ready >= submitted) break;
                io.futexWaitUncancelable(u32, &self.persistent_ready_count.raw, ready);
            }

            if (self.first_initialized_io == .none) self.first_initialized_io = .persistent;
            self.persistent_init_state.store(2, .release);
            return true;
        }

        while (true) {
            const current = self.persistent_init_state.load(.acquire);
            if (current == 2) return true;
            if (current == 3) return false;
            std.atomic.spinLoopHint();
        }
    }

    pub fn taskCount(self: *const Executor, items: usize, min_items_per_task: usize) usize {
        const limit = self.policy.max_threads();
        if (limit <= 1 or items < min_items_per_task * 2) return 1;
        const by_size = @max(@as(usize, 1), items / min_items_per_task);
        return @min(@min(limit, max_tasks), by_size);
    }

    fn runPersistent(self: *Executor, cache_domain: ?*const WorkerCacheDomain, task_fn: TaskFn, tasks: *const anyopaque, count: usize) bool {
        if (!self.ensurePersistentWorkers()) return false;
        const workers = self.persistent_worker_count.load(.acquire);
        if (workers == 0) return false;
        const task_helpers: u32 = @intCast(count - 1);
        const allowed_helpers: u32 = @intCast(self.configuredHelperCount());
        if (workers < task_helpers or allowed_helpers < task_helpers) return false;
        const active_helpers = task_helpers;
        if (active_helpers == 0) return false;
        const first_helper: u32 = if (count == 3 and workers >= active_helpers + 2) 2 else 0;

        self.persistent_cache_domain = cache_domain;
        self.persistent_task_fn = task_fn;
        self.persistent_tasks = tasks;
        self.persistent_job_count.store(@intCast(count), .release);
        self.persistent_done_count.store(0, .release);
        self.persistent_done_target.store(active_helpers, .release);
        self.persistent_first_helper.store(first_helper, .release);
        self.persistent_active_helpers.store(active_helpers, .release);

        const io = self.persistent_threaded.io();
        // Wake sleeping macOS subset helpers before the caller starts its tile,
        // but avoid a syscall for hot helpers. This flag is only a latency hint:
        // the unconditional late wake below also covers a stale or racing hint.
        const wake_waiting_helpers = @import("builtin").target.os.tag == .macos and
            (first_helper != 0 or active_helpers != workers);
        for (0..active_helpers) |worker_id| {
            const target_worker = first_helper + worker_id;
            // self.io_busy serializes jobs, and completion acknowledges each worker's
            // observed generation before another job can be submitted. Increment
            // locally so even wraparound differs from an idle worker's last value.
            // The release publishes the task payload to the worker's acquire load.
            _ = self.persistent_worker_generation[target_worker].fetchAdd(1, .release);
            if (wake_waiting_helpers and self.persistent_worker_waiting[target_worker].load(.acquire) != 0)
                io.futexWake(u32, &self.persistent_worker_generation[target_worker].raw, 1);
        }

        self.policy.configure_thread(null);
        task_fn(tasks, 0);
        var woke_sleepers = false;
        while (true) {
            var done = self.persistent_done_count.load(.acquire);
            const done_target = self.persistent_done_target.load(.acquire);
            var spin_count: u32 = 0;
            while (done < done_target and spin_count < persistent_done_spin_iters) : (spin_count += 1) {
                std.atomic.spinLoopHint();
                done = self.persistent_done_count.load(.acquire);
            }
            if (done >= done_target) break;
            if (!woke_sleepers) {
                for (0..active_helpers) |worker_id| {
                    io.futexWake(u32, &self.persistent_worker_generation[first_helper + worker_id].raw, 1);
                }
                woke_sleepers = true;
                done = self.persistent_done_count.load(.acquire);
                if (done >= done_target) break;
            }
            io.futexWaitUncancelable(u32, &self.persistent_done_count.raw, done);
        }
        return true;
    }

    fn runGroup(self: *Executor, cache_domain: ?*const WorkerCacheDomain, task_fn: TaskFn, tasks: *const anyopaque, count: usize) bool {
        if (!self.ensureIoThreaded()) return false;
        if (count - 1 > self.configuredHelperCount()) return false;

        var group: std.Io.Group = .init;
        var submitted = false;
        const io = self.io_threaded.io();
        for (1..count) |index| {
            group.concurrent(io, runIoTask, .{ self, cache_domain, task_fn, tasks, index }) catch {
                task_fn(tasks, index);
                continue;
            };
            submitted = true;
        }
        self.policy.configure_thread(null);
        task_fn(tasks, 0);
        if (submitted) group.await(io) catch {};
        return true;
    }

    pub fn run(self: *Executor, batch: Batch) bool {
        const task_fn = batch.task_fn;
        const tasks = batch.tasks;
        const count = batch.count;
        if (count <= 1) return false;
        if (self.io_busy.cmpxchgStrong(0, 1, .acq_rel, .acquire) != null) return false;
        defer self.io_busy.store(0, .release);

        return self.runGroup(batch.cache_domain, task_fn, tasks, count);
    }

    pub fn runLowLatency(self: *Executor, batch: Batch) bool {
        const task_fn = batch.task_fn;
        const tasks = batch.tasks;
        const count = batch.count;
        if (count <= 1) return false;
        if (self.io_busy.cmpxchgStrong(0, 1, .acq_rel, .acquire) != null) return false;
        defer self.io_busy.store(0, .release);

        if (self.runPersistent(batch.cache_domain, task_fn, tasks, count)) return true;
        return self.runGroup(batch.cache_domain, task_fn, tasks, count);
    }

    /// Distribute independent output regions without a barrier between tiles.
    /// A false return guarantees that no job has modified caller output.
    pub fn runQueued(self: *Executor, batch: Batch, worker_limit: usize) bool {
        const task_fn = batch.task_fn;
        const tasks = batch.tasks;
        const count = batch.count;
        const workers = @min(count, @min(worker_limit, self.configuredHelperCount() + 1));
        if (workers <= 1) return false;
        const Queue = struct {
            next: std.atomic.Value(usize) = .init(0),
            count: usize,
            task_fn: TaskFn,
            tasks: *const anyopaque,

            fn execute(raw: *const anyopaque, _: usize) void {
                const queue: *@This() = @ptrCast(@alignCast(@constCast(raw)));
                while (true) {
                    const index = queue.next.fetchAdd(1, .monotonic);
                    if (index >= queue.count) return;
                    queue.task_fn(queue.tasks, index);
                }
            }
        };
        var queue = Queue{ .count = count, .task_fn = task_fn, .tasks = tasks };
        return self.runLowLatency(.{ .task_fn = Queue.execute, .tasks = &queue, .count = workers, .cache_domain = batch.cache_domain });
    }

    fn waitForInitState(state: *std.atomic.Value(u8)) u8 {
        while (true) {
            const current = state.load(.acquire);
            if (current != 1) return current;
            std.atomic.spinLoopHint();
        }
    }

    fn shutdownPersistentLocked(self: *Executor) void {
        const state = waitForInitState(&self.persistent_init_state);
        if (state == 3) {
            self.persistent_init_state.store(0, .release);
            return;
        }
        if (state != 2) return;

        const worker_count = self.persistent_worker_count.load(.acquire);
        if (worker_count != 0) {
            const io = self.persistent_threaded.io();
            self.persistent_shutdown_requested.store(1, .release);
            self.persistent_exited_count.store(0, .release);

            for (0..worker_count) |worker_id| {
                _ = self.persistent_worker_generation[worker_id].fetchAdd(1, .release);
                io.futexWake(u32, &self.persistent_worker_generation[worker_id].raw, 1);
            }

            while (true) {
                const exited = self.persistent_exited_count.load(.acquire);
                if (exited >= worker_count) break;
                io.futexWaitUncancelable(u32, &self.persistent_exited_count.raw, exited);
            }
            self.persistent_group.await(io) catch {};
        }

        self.persistent_threaded.deinit();
        self.persistent_group = .init;
        self.persistent_worker_count.store(0, .release);
        self.persistent_ready_count.store(0, .release);
        self.persistent_exited_count.store(0, .release);
        self.persistent_shutdown_requested.store(0, .release);
        self.persistent_init_state.store(0, .release);
    }

    fn shutdownIoThreadedLocked(self: *Executor) void {
        const state = waitForInitState(&self.io_init_state);
        if (state == 2) self.io_threaded.deinit();
        if (state == 2 or state == 3) self.io_init_state.store(0, .release);
    }

    pub fn drainAndReset(self: *Executor) void {
        while (self.io_busy.cmpxchgStrong(0, 1, .acq_rel, .acquire) != null) {
            std.atomic.spinLoopHint();
        }
        defer self.io_busy.store(0, .release);

        if (self.first_initialized_io == .persistent) {
            self.shutdownIoThreadedLocked();
            self.shutdownPersistentLocked();
        } else {
            self.shutdownPersistentLocked();
            self.shutdownIoThreadedLocked();
        }
        self.first_initialized_io = .none;
    }
};

test "queued jobs execute exactly once and refusal leaves output untouched" {
    const fixture = @import("test_support.zig");
    var executor = Executor.init(std.heap.c_allocator, fixture.policy);
    const Jobs = struct {
        counters: [257]std.atomic.Value(u32) = @as([257]std.atomic.Value(u32), @splat(.init(0))),
        fn execute(raw: *const anyopaque, index: usize) void {
            const self: *@This() = @ptrCast(@alignCast(@constCast(raw)));
            _ = self.counters[index].fetchAdd(1, .monotonic);
        }
    };
    var jobs: Jobs = .{};
    fixture.setThreads(4);
    defer {
        executor.drainAndReset();
        fixture.setThreads(0);
    }
    try std.testing.expect(!executor.runQueued(.{ .task_fn = Jobs.execute, .tasks = &jobs, .count = 257 }, 1));
    executor.io_busy.store(1, .release);
    const busy_result = executor.runQueued(.{ .task_fn = Jobs.execute, .tasks = &jobs, .count = 257 }, 4);
    executor.io_busy.store(0, .release);
    try std.testing.expect(!busy_result);
    for (&jobs.counters) |*counter| try std.testing.expectEqual(@as(u32, 0), counter.load(.monotonic));
    if (executor.configuredHelperCount() == 0) return;
    for (1..5) |iteration| {
        try std.testing.expect(executor.runQueued(.{ .task_fn = Jobs.execute, .tasks = &jobs, .count = 257 }, 3));
        for (&jobs.counters) |*counter| try std.testing.expectEqual(@as(u32, @intCast(iteration)), counter.load(.monotonic));
    }
}

test "runLowLatency refuses partial execution when helpers cannot cover tasks" {
    const fixture = @import("test_support.zig");
    var executor = Executor.init(std.heap.c_allocator, fixture.policy);
    const CounterTask = struct {
        counter: *std.atomic.Value(u32),

        fn run(raw_tasks: *const anyopaque, index: usize) void {
            const tasks: [*]const @This() = @ptrCast(@alignCast(raw_tasks));
            _ = tasks[index].counter.fetchAdd(1, .monotonic);
        }
    };

    var counter = std.atomic.Value(u32).init(0);
    var tasks = [_]CounterTask{
        .{ .counter = &counter },
        .{ .counter = &counter },
        .{ .counter = &counter },
        .{ .counter = &counter },
    };

    fixture.setThreads(2);
    defer {
        fixture.setThreads(0);
        executor.drainAndReset();
    }

    try std.testing.expect(!executor.runLowLatency(.{ .task_fn = CounterTask.run, .tasks = @ptrCast(&tasks), .count = tasks.len }));
    try std.testing.expectEqual(@as(u32, 0), counter.load(.monotonic));

    executor.drainAndReset();
    fixture.setThreads(5);
    if (executor.configuredHelperCount() == 0) return;

    // Seed only while stopped: workers must acquire the seed before announcing
    // readiness. The first phase wraps on shutdown, the second on submission.
    for ([_]u32{ std.math.maxInt(u32) - 1, std.math.maxInt(u32) }) |seed| {
        for (&executor.persistent_worker_generation) |*generation| generation.store(seed, .release);
        try std.testing.expect(executor.ensurePersistentWorkers());
        const workers = executor.persistent_worker_count.load(.acquire);
        try std.testing.expect(workers > 0);
        var counters: [5]std.atomic.Value(u32) = @splat(std.atomic.Value(u32).init(0));
        var distinct_tasks: [5]CounterTask = undefined;
        for (&distinct_tasks, &counters) |*task, *task_counter| task.* = .{ .counter = task_counter };

        const counts = [_]usize{ 2, @min(workers + 1, 3), 2, workers + 1 };
        const runs: usize = if (seed == std.math.maxInt(u32) - 1) 1 else counts.len;
        for (counts[0..runs]) |count| {
            // Exercise a sleeping subset whose first helper is not zero,
            // as well as the hot generation-wraparound path.
            if (count == 3 and workers >= 4) try std.Io.sleep(executor.persistent_threaded.io(), .fromMilliseconds(100), .awake);
            var before: [max_tasks]u32 = undefined;
            for (0..workers) |worker_id| before[worker_id] = executor.persistent_worker_generation[worker_id].load(.acquire);
            for (&counters) |*task_counter| task_counter.store(0, .monotonic);
            try std.testing.expect(executor.runLowLatency(.{ .task_fn = CounterTask.run, .tasks = @ptrCast(&distinct_tasks), .count = count }));
            // Separate counters catch a duplicated index hiding a missing one.
            for (&counters, 0..) |*task_counter, index| {
                try std.testing.expectEqual(@as(u32, if (index < count) 1 else 0), task_counter.load(.monotonic));
            }
            const active_helpers = count - 1;
            const first_helper: usize = if (count == 3 and workers >= active_helpers + 2) 2 else 0;
            for (0..workers) |worker_id| {
                const active = worker_id >= first_helper and worker_id < first_helper + active_helpers;
                const expected = before[worker_id] +% @as(u32, if (active) 1 else 0);
                try std.testing.expectEqual(expected, executor.persistent_worker_generation[worker_id].load(.acquire));
            }
        }

        var before_shutdown: [max_tasks]u32 = undefined;
        for (0..workers) |worker_id| before_shutdown[worker_id] = executor.persistent_worker_generation[worker_id].load(.acquire);
        executor.drainAndReset();
        try std.testing.expectEqual(@as(u8, 0), executor.persistent_init_state.load(.acquire));
        try std.testing.expectEqual(@as(u32, 0), executor.persistent_worker_count.load(.acquire));
        for (0..workers) |worker_id| {
            try std.testing.expectEqual(before_shutdown[worker_id] +% 1, executor.persistent_worker_generation[worker_id].load(.acquire));
        }
    }
}

const Counter = struct {
    visits: [8]std.atomic.Value(u32) = @splat(.init(0)),
    helper_seen: std.atomic.Value(bool) = .init(false),
    caller: std.Thread.Id = undefined,
    fn execute(raw: *const anyopaque, index: usize) void {
        const self: *@This() = @ptrCast(@alignCast(@constCast(raw)));
        _ = self.visits[index].fetchAdd(1, .monotonic);
        if (index != 0 and std.Thread.getCurrentId() != self.caller) self.helper_seen.store(true, .release);
    }
    fn batch(self: *@This(), count: usize) Batch {
        return .{ .task_fn = execute, .tasks = self, .count = count };
    }
};

test "real persistent zero and partial submissions release captured allocator and restart" {
    const fixture = @import("test_support.zig");
    fixture.setThreads(4);
    var allocations: fixture.Allocations = .{ .budget = 0 };
    var executor = Executor.init(allocations.allocator(), fixture.policy);
    defer executor.drainAndReset();
    var work: Counter = .{ .caller = std.Thread.getCurrentId() };
    try std.testing.expect(!executor.runPersistent(null, Counter.execute, &work, 2));
    try std.testing.expectEqual(@as(u8, 3), executor.persistent_init_state.load(.acquire));
    try std.testing.expectEqual(Executor.FirstInitializedIo.none, executor.first_initialized_io);
    try std.testing.expectEqual(@as(usize, 1), allocations.stats().rejected);
    executor.drainAndReset();
    try std.testing.expectEqual(@as(u8, 0), executor.persistent_init_state.load(.acquire));
    // The same captured provider is now allowed one real task allocation.
    allocations.budget = 1;
    try std.testing.expect(executor.runPersistent(null, Counter.execute, &work, 2));
    try std.testing.expect(work.helper_seen.load(.acquire));
    try std.testing.expectEqual(@as(u32, 1), executor.persistent_worker_count.load(.acquire));
    try std.testing.expectEqual(@as(u32, 1), executor.persistent_ready_count.load(.acquire));
    try std.testing.expectEqual(@as(usize, 1), allocations.stats().accepted);
    try std.testing.expectEqual(@as(usize, 2), allocations.stats().rejected);
    var refused: Counter = .{ .caller = std.Thread.getCurrentId() };
    try std.testing.expect(!executor.runPersistent(null, Counter.execute, &refused, 3));
    for (&refused.visits) |*v| try std.testing.expectEqual(@as(u32, 0), v.load(.acquire));
    // Ordinary fallback is real, but its allocations also fail: true + inline.
    try std.testing.expect(executor.runLowLatency(refused.batch(3)));
    for (refused.visits[0..3]) |v| try std.testing.expectEqual(@as(u32, 1), v.load(.acquire));
    try std.testing.expect(!refused.helper_seen.load(.acquire));
    executor.drainAndReset();
    try std.testing.expectEqual(allocations.stats().accepted, allocations.stats().freed);
    try std.testing.expectEqual(@as(usize, 1), allocations.stats().freed);
}

test "ordinary real zero and partial submission execute all indices exactly once" {
    const fixture = @import("test_support.zig");
    fixture.setThreads(4);
    for ([_]usize{ 0, 1 }) |budget| {
        var allocations: fixture.Allocations = .{ .budget = budget };
        var executor = Executor.init(allocations.allocator(), fixture.policy);
        defer executor.drainAndReset();
        var work: Counter = .{ .caller = std.Thread.getCurrentId() };
        try std.testing.expect(executor.run(work.batch(4)));
        for (work.visits[0..4]) |v| try std.testing.expectEqual(@as(u32, 1), v.load(.acquire));
        try std.testing.expectEqual(budget == 1, work.helper_seen.load(.acquire));
        executor.drainAndReset();
        const stats = allocations.stats();
        try std.testing.expectEqual(@as(usize, 3), stats.attempts);
        try std.testing.expectEqual(budget, stats.accepted);
        try std.testing.expectEqual(3 - budget, stats.rejected);
        try std.testing.expectEqual(budget, stats.freed);
    }
}

const Simultaneous = struct {
    executor: *Executor,
    entered: std.atomic.Value(bool) = .init(false),
    release: std.atomic.Value(bool) = .init(false),
    callback_done: std.atomic.Value(bool) = .init(false),
    reset_started: std.atomic.Value(bool) = .init(false),
    reset_done: std.atomic.Value(bool) = .init(false),
    reset_saw_done: std.atomic.Value(bool) = .init(false),
    accepted: bool = false,
    visits: [2]std.atomic.Value(u32) = @splat(.init(0)),
    fn wait(flag: *const std.atomic.Value(bool)) void {
        while (!flag.load(.acquire)) std.Thread.yield() catch {};
    }
    fn execute(raw: *const anyopaque, index: usize) void {
        const self: *@This() = @ptrCast(@alignCast(@constCast(raw)));
        _ = self.visits[index].fetchAdd(1, .monotonic);
        if (index == 0) {
            self.entered.store(true, .release);
            wait(&self.release);
            self.callback_done.store(true, .release);
        }
    }
    fn run(self: *@This()) void {
        self.accepted = self.executor.runLowLatency(.{ .task_fn = execute, .tasks = self, .count = 2 });
    }
    fn reset(self: *@This()) void {
        self.reset_started.store(true, .release);
        self.executor.drainAndReset();
        self.reset_saw_done.store(self.callback_done.load(.acquire), .release);
        self.reset_done.store(true, .release);
    }
};

test "simultaneous caller refusal and reset preserve synchronous borrowed lifetime" {
    const fixture = @import("test_support.zig");
    fixture.setThreads(2);
    var executor = Executor.init(std.heap.c_allocator, fixture.policy);
    defer executor.drainAndReset();
    var active: Simultaneous = .{ .executor = &executor };
    const first = try std.Thread.spawn(.{}, Simultaneous.run, .{&active});
    // Always release/join an accepted batch if any assertion fails below.
    defer {
        active.release.store(true, .release);
        first.join();
    }
    Simultaneous.wait(&active.entered);
    var rejected: Counter = .{ .caller = std.Thread.getCurrentId() };
    try std.testing.expect(!executor.run(rejected.batch(2)));
    try std.testing.expect(!executor.runLowLatency(rejected.batch(2)));
    try std.testing.expect(!executor.runQueued(rejected.batch(8), 2));
    for (&rejected.visits) |*v| try std.testing.expectEqual(@as(u32, 0), v.load(.acquire));
    // Caller B owns this fallback, outside the executor's drain contract.
    Counter.execute(&rejected, 0);
    Counter.execute(&rejected, 1);
    const closer = try std.Thread.spawn(.{}, Simultaneous.reset, .{&active});
    Simultaneous.wait(&active.reset_started);
    active.release.store(true, .release);
    closer.join();
    try std.testing.expect(active.reset_done.load(.acquire));
    try std.testing.expect(active.reset_saw_done.load(.acquire));
    for (&active.visits) |*v| try std.testing.expectEqual(@as(u32, 1), v.load(.acquire));
    for (rejected.visits[0..2]) |v| try std.testing.expectEqual(@as(u32, 1), v.load(.acquire));
    // A fresh batch after reverse close reinitializes and completes.
    var restarted: Counter = .{ .caller = std.Thread.getCurrentId() };
    try std.testing.expect(executor.run(restarted.batch(2)));
}

const CacheCheck = struct {
    threadlocal var marker: u8 = 0;
    threadlocal var producer: std.Thread.Id = undefined;
    var caller: std.Thread.Id = undefined;
    var errors: std.atomic.Value(u32) = .init(0);
    var released_a: std.atomic.Value(u32) = .init(0);
    var released_b: std.atomic.Value(u32) = .init(0);
    const domain_a: WorkerCacheDomain = .{ .release_current_worker = releaseA };
    const domain_b: WorkerCacheDomain = .{ .release_current_worker = releaseB };
    fn releaseA() void {
        release(1);
    }
    fn releaseB() void {
        release(2);
    }
    fn release(tag: u8) void {
        if (marker != tag or producer != std.Thread.getCurrentId() or caller == std.Thread.getCurrentId()) _ = errors.fetchAdd(1, .monotonic);
        marker = 0;
        if (tag == 1) {
            _ = released_a.fetchAdd(1, .monotonic);
        } else {
            _ = released_b.fetchAdd(1, .monotonic);
        }
    }
    fn execute(raw: *const anyopaque, index: usize) void {
        const tag: *const u8 = @ptrCast(raw);
        if (index == 0 or caller == std.Thread.getCurrentId()) {
            if (marker != 99) _ = errors.fetchAdd(1, .monotonic);
            return;
        }
        if (marker != 0 and marker != tag.*) _ = errors.fetchAdd(1, .monotonic);
        if (tag.* == 0 and marker != 0) _ = errors.fetchAdd(1, .monotonic);
        marker = tag.*;
        producer = std.Thread.getCurrentId();
    }
    fn reset() void {
        caller = std.Thread.getCurrentId();
        marker = 99;
        errors.store(0, .release);
        released_a.store(0, .release);
        released_b.store(0, .release);
    }
};

test "persistent domains same different null subset release on their own worker" {
    const fixture = @import("test_support.zig");
    fixture.setThreads(4);
    var executor = Executor.init(std.heap.c_allocator, fixture.policy);
    defer executor.drainAndReset();
    CacheCheck.reset();
    const tags = [_]u8{ 1, 1, 0, 2, 1 };
    const counts = [_]usize{ 4, 2, 2, 4, 4 };
    const domains = [_]?*const WorkerCacheDomain{ &CacheCheck.domain_a, &CacheCheck.domain_a, null, &CacheCheck.domain_b, &CacheCheck.domain_a };
    for (&tags, counts, domains) |*tag, count, domain| {
        try std.testing.expect(executor.runPersistent(domain, CacheCheck.execute, tag, count));
    }
    try std.testing.expectEqual(@as(u32, 3), CacheCheck.released_a.load(.acquire));
    try std.testing.expectEqual(@as(u32, 3), CacheCheck.released_b.load(.acquire));
    executor.drainAndReset();
    try std.testing.expectEqual(@as(u32, 6), CacheCheck.released_a.load(.acquire));
    try std.testing.expectEqual(@as(u32, 0), CacheCheck.errors.load(.acquire));
    try std.testing.expectEqual(@as(u8, 99), CacheCheck.marker);
}

test "ordinary helper cleanup excludes caller and real inline allocation fallbacks" {
    const fixture = @import("test_support.zig");
    fixture.setThreads(4);
    for ([_]usize{ 0, 1, 3 }) |budget| {
        var allocations: fixture.Allocations = .{ .budget = budget };
        var executor = Executor.init(allocations.allocator(), fixture.policy);
        defer executor.drainAndReset();
        CacheCheck.reset();
        const tag: u8 = 1;
        try std.testing.expect(executor.run(.{ .task_fn = CacheCheck.execute, .tasks = &tag, .count = 4, .cache_domain = &CacheCheck.domain_a }));
        executor.drainAndReset();
        try std.testing.expectEqual(@as(u32, @intCast(budget)), CacheCheck.released_a.load(.acquire));
        try std.testing.expectEqual(@as(u32, 0), CacheCheck.errors.load(.acquire));
        try std.testing.expectEqual(@as(u8, 99), CacheCheck.marker);
        try std.testing.expectEqual(allocations.stats().accepted, allocations.stats().freed);
    }
}

// Explicitly discover the independent common lifetime tests in this existing
// contract root; named dependencies alone do not enumerate their tests.
comptime {
    if (@import("builtin").is_test) _ = @import("context_test.zig");
}

comptime {
    if (@import("builtin").is_test) _ = @import("context_scratch_test.zig");
}
