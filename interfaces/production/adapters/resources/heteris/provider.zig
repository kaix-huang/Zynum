// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
const std = @import("std");
pub const c = @import("heteris-provider-c");
const execution = @import("zynum-execution");
const results = execution.results;
const ports = execution.resource_port;
const contracts = execution.contracts;
const CallReport = results.CallReport;
const ExecutionError = results.ExecutionError;

pub fn failStop(report: *const CallReport, reason: []const u8) noreturn {
    std.debug.panic("HETERIS_PROVIDER_CONTRACT_FAILURE: {s}; stage={d} status={d} execution={d} callback={d} cleanup={d}", .{ reason, @backingInt(report.provider.stage), report.provider.returned_status, report.provider.execution_status, report.provider.callback_status, report.provider.cleanup_status });
}
pub fn mapStatus(code: i32) ExecutionError {
    return switch (code) {
        c.HETERIS_BUDGET_EXCEEDED => error.ResourceBudgetExceeded,
        c.HETERIS_OUT_OF_MEMORY => error.OutOfMemory,
        c.HETERIS_BUSY => error.ProviderBusy,
        c.HETERIS_CLOSED => error.ProviderClosed,
        c.HETERIS_WOULD_DEADLOCK => error.WouldDeadlock,
        c.HETERIS_UNSUPPORTED, c.HETERIS_UNAVAILABLE => error.Unsupported,
        else => error.ProviderRejected,
    };
}
fn tokenValid(token: c.heteris_rp_token, authority: ?*anyopaque, kind: u32) bool {
    return authority != null and token.authority == authority and token.serial != 0 and token.generation == 1 and token.kind == kind and token.reserved == 0;
}
fn identityValid(id: c.heteris_rp_identity) bool {
    return id.serial != 0 and id.generation == 1;
}
fn sameIdentity(a: c.heteris_rp_identity, b: c.heteris_rp_identity) bool {
    return a.serial == b.serial and a.generation == b.generation;
}

/// Copies the complete required immutable v1 prefix. Never retains caller
/// descriptor/table storage or asks for root destruction/drain/reset rights.
pub const Binding = struct {
    api: c.heteris_runtime_provider_api_v1,
    token: c.heteris_rp_token,
    caps: c.heteris_rp_cpu_caps_v1,
    parallel_limit: u32,

    pub fn init(provider: *const c.heteris_runtime_provider_v1, desc: results.ContextDesc, report: *CallReport) ExecutionError!Binding {
        report.* = .{};
        report.provider.stage = .descriptor;
        const source = provider.*;
        if (source.abi_version != 1 or source.struct_size < @sizeOf(c.heteris_runtime_provider_v1) or source.api == null) return error.InvalidProvider;
        if (@intFromPtr(source.api) % @alignOf(c.heteris_runtime_provider_api_v1) != 0) return error.InvalidProvider;
        if (source.api[0].abi_version != 1 or source.api[0].struct_size < @sizeOf(c.heteris_runtime_provider_api_v1)) return error.InvalidProvider;
        const api = source.api[0];
        if (!tokenValid(source.provider, api.authority, c.HETERIS_RP_PROVIDER)) return error.InvalidProvider;
        if (api.provider_retain == null or api.provider_release == null or api.cpu_query == null or api.same_memory_domain == null or api.scratch_allocate == null or api.scratch_retain == null or api.scratch_release == null or api.cpu_bulk_sync == null) return error.InvalidProvider;
        if (desc.algorithm != .portable_generic) return error.Unsupported;
        report.provider.stage = .retain;
        const retained = api.provider_retain.?(source.provider);
        report.provider.returned_status = retained;
        if (retained != c.HETERIS_OK) return mapStatus(retained);
        errdefer {
            const cleanup = api.provider_release.?(source.provider);
            if (cleanup != c.HETERIS_OK) {
                report.provider.cleanup_status = cleanup;
                report.provider.stage = .rollback_release;
                failStop(report, "initialization rollback release failed");
            }
        }
        var caps: c.heteris_rp_cpu_caps_v1 = std.mem.zeroes(c.heteris_rp_cpu_caps_v1);
        report.provider.stage = .query;
        const queried = api.cpu_query.?(source.provider, &caps);
        report.provider.returned_status = queried;
        if (queried != c.HETERIS_OK) return mapStatus(queried);
        if (caps.abi_version != 1 or caps.struct_size < @sizeOf(@TypeOf(caps)) or caps.reserved != 0 or !identityValid(caps.provider_instance) or !identityValid(caps.runtime_instance) or !identityValid(caps.memory_domain) or caps.provider_instance.serial != source.provider.serial or caps.provider_instance.generation != source.provider.generation) return error.InvalidProvider;
        const flags = c.HETERIS_RP_CPU_BULK_SYNC | c.HETERIS_RP_HOST_SCRATCH | c.HETERIS_RP_EXPLICIT_HOST_BACKING_BUDGET | c.HETERIS_RP_ORIGINAL_MANAGED_THREAD_BUDGET | c.HETERIS_RP_QUIESCENT_RETURN;
        if (caps.mode != c.HETERIS_RESOURCE_MODE_INTEGRATED or caps.device_kind != c.HETERIS_DEVICE_CPU or caps.flags & flags != flags) return error.Unsupported;
        const managed = std.math.add(u32, caps.compute_workers, 1) catch return error.InvalidProvider;
        if (caps.compute_workers == 0 or caps.managed_threads < managed or caps.max_bulk_commands == 0 or caps.max_bulk_commands > caps.compute_workers) return error.InvalidProvider;
        const limit = @min(caps.compute_workers, caps.max_bulk_commands, contracts.max_tasks);
        if (desc.max_parallel_tasks > limit) return error.Unsupported;
        return .{ .api = api, .token = source.provider, .caps = caps, .parallel_limit = if (desc.max_parallel_tasks == 0) @intCast(limit) else desc.max_parallel_tasks };
    }
    pub fn release(self: *const Binding) i32 {
        return self.api.provider_release.?(self.token);
    }
    pub fn port(self: *Binding) ports.ResourcePort {
        return .{ .owner = self, .vtable = &vtable, .parallel_limit = self.parallel_limit };
    }
    fn acquireBytes(self: *const Binding, bytes: usize, alignment: usize, report: *CallReport) ExecutionError!ports.ByteLease {
        const bytes64 = std.math.cast(u64, bytes) orelse return error.ExtentOverflow;
        _ = std.math.add(usize, bytes, alignment - 1) catch return error.ExtentOverflow;
        report.provider.stage = .scratch_allocate;
        var scratch: c.heteris_rp_scratch_v1 = std.mem.zeroes(c.heteris_rp_scratch_v1);
        const code = self.api.scratch_allocate.?(self.token, bytes64, alignment, &scratch);
        report.provider.returned_status = code;
        if (code != c.HETERIS_OK) return mapStatus(code);
        const charged = std.math.add(u64, bytes64, alignment - 1) catch failStop(report, "scratch charge overflow");
        if (scratch.abi_version != 1 or scratch.struct_size < @sizeOf(@TypeOf(scratch)) or !tokenValid(scratch.lease, self.api.authority, c.HETERIS_RP_SCRATCH) or !sameIdentity(scratch.memory_domain, self.caps.memory_domain) or scratch.data == null or scratch.logical_bytes != bytes64 or scratch.alignment != alignment or scratch.accounted_bytes != charged) failStop(report, "malformed scratch ownership descriptor");
        const address = @intFromPtr(scratch.data);
        if (address % alignment != 0) failStop(report, "misaligned scratch backing");
        _ = std.math.add(usize, address, bytes) catch failStop(report, "scratch address extent overflow");
        const pointer: [*]u8 = @ptrCast(scratch.data.?);
        return .{ .values = pointer[0..bytes], .alignment = alignment, .ownership = ports.Ownership.init(LeaseHandle, .{ .token = scratch.lease, .release_fn = self.api.scratch_release.? }), .release_fn = releaseScratch };
    }
    fn runSync(self: *const Binding, batch: *const ports.TaskBatch, report: *CallReport) ExecutionError!void {
        const count64 = std.math.cast(u64, batch.count) orelse return error.ExtentOverflow;
        var outcome: c.heteris_rp_bulk_result_v1 = std.mem.zeroes(c.heteris_rp_bulk_result_v1);
        report.provider.stage = .submit;
        const code = self.api.cpu_bulk_sync.?(self.token, 0, count64, 1, self.parallel_limit, bulkRange, @constCast(batch), &outcome);
        report.provider.returned_status = code;
        report.provider.execution_status = outcome.execution_status;
        report.provider.callback_status = outcome.callback_status;
        report.provider.callbacks_started = outcome.callbacks_started;
        report.provider.callbacks_completed = outcome.callbacks_completed;
        // Preserve every raw result field even when the provider completion is malformed.
        if (outcome.abi_version != 1 or outcome.struct_size < @sizeOf(@TypeOf(outcome)) or outcome.reserved != 0 or outcome.callbacks_started > count64 or outcome.callbacks_completed > outcome.callbacks_started) failStop(report, "malformed bulk completion prefix/counts");
        const expected = if (outcome.execution_status != c.HETERIS_OK) outcome.execution_status else outcome.callback_status;
        if (code != expected) failStop(report, "bulk return/status mismatch");
        switch (outcome.state) {
            c.HETERIS_RP_REJECTED => {
                report.provider.state = .rejected;
                if (code == c.HETERIS_OK or outcome.callbacks_started != 0 or outcome.callbacks_completed != 0 or outcome.callback_status != c.HETERIS_OK) failStop(report, "rejected bulk performed callbacks");
                return mapStatus(code);
            },
            c.HETERIS_RP_QUIESCENT => {
                report.provider.state = .quiescent;
                report.output = .partial_or_invalid;
                if (outcome.callbacks_started != outcome.callbacks_completed) failStop(report, "bulk returned before all callbacks completed");
                if (outcome.execution_status != c.HETERIS_OK) return error.ProviderExecutionFailed;
                if (outcome.callback_status != c.HETERIS_OK) return error.ProviderCallbackFailed;
                if (outcome.callbacks_completed != count64) failStop(report, "successful bulk did not cover every task");
                report.output = .complete;
            },
            else => failStop(report, "bulk returned unknown access state"),
        }
    }
};

/// Ordinary runtime checks are deliberately present in ReleaseFast too.
fn bulkRange(data: ?*anyopaque, begin: u64, end: u64) callconv(.c) i32 {
    const raw = data orelse return c.HETERIS_INVALID_ARGUMENT;
    if (@intFromPtr(raw) % @alignOf(ports.TaskBatch) != 0) return c.HETERIS_INVALID_ARGUMENT;
    const invocation: *const ports.TaskBatch = @ptrCast(@alignCast(raw));
    if (begin > end) return c.HETERIS_INVALID_ARGUMENT;
    if (end > invocation.count) return c.HETERIS_OUT_OF_BOUNDS;
    const first = std.math.cast(usize, begin) orelse return c.HETERIS_OVERFLOW;
    const last = std.math.cast(usize, end) orelse return c.HETERIS_OVERFLOW;
    for (first..last) |index| {
        if (index >= invocation.count) return c.HETERIS_OUT_OF_BOUNDS;
        invocation.task_fn(invocation.tasks, index);
    }
    return c.HETERIS_OK;
}

const LeaseHandle = struct { token: c.heteris_rp_token, release_fn: *const fn (c.heteris_rp_token) callconv(.c) c.heteris_status };
fn releaseScratch(storage: *const ports.Ownership, report: *CallReport) void {
    const handle = storage.value(LeaseHandle);
    const code = handle.release_fn(handle.token);
    if (code != c.HETERIS_OK) {
        report.provider.stage = .scratch_release;
        report.provider.cleanup_status = code;
        failStop(report, "valid scratch release failed");
    }
}
fn cast(raw: *anyopaque) *Binding {
    return @ptrCast(@alignCast(raw));
}
fn acquireBytes(raw: *anyopaque, bytes: usize, alignment: usize, report: *CallReport) ExecutionError!ports.ByteLease {
    return cast(raw).acquireBytes(bytes, alignment, report);
}
fn runSync(raw: *anyopaque, batch: *const ports.TaskBatch, report: *CallReport) ExecutionError!void {
    return cast(raw).runSync(batch, report);
}
fn releaseProvider(raw: *anyopaque, report: *CallReport) bool {
    const code = cast(raw).release();
    report.provider.stage = .provider_release;
    report.provider.returned_status = code;
    report.provider.cleanup_status = if (code == c.HETERIS_OK) 0 else code;
    return code == c.HETERIS_OK;
}
const vtable: ports.VTable = .{ .acquire_bytes = acquireBytes, .run_sync = runSync, .release_provider = releaseProvider };

/// Caller-owned production envelope. Initialize only at its final stable address;
/// do not copy or move it while live. No temporary owner or hidden allocation.
pub const IntegratedContext = struct {
    context: execution.Context,
    binding: Binding,
    pub fn initInPlace(self: *IntegratedContext, source: *const c.heteris_runtime_provider_v1, desc: results.ContextDesc, report: *CallReport) ExecutionError!void {
        self.binding = try Binding.init(source, desc, report);
        self.context = execution.Context.initResources(self.binding.port());
    }
    pub fn core(self: *IntegratedContext) *execution.Context {
        return &self.context;
    }
    pub fn requestClose(self: *IntegratedContext) execution.RequestResult {
        return self.context.requestClose();
    }
    pub fn tryProgressClose(self: *IntegratedContext, report: *CallReport) execution.CloseResult {
        return self.context.tryProgressClose(report);
    }
};

test "bulk range rejects malformed bounds before reading tasks" {
    const Test = struct {
        fn call(_: *const anyopaque, _: usize) void {
            @panic("malformed range reached task");
        }
    };
    var invocation: ports.TaskBatch = .{ .task_fn = Test.call, .tasks = undefined, .count = 3 };
    try std.testing.expectEqual(@as(i32, c.HETERIS_INVALID_ARGUMENT), bulkRange(null, 0, 1));
    try std.testing.expectEqual(@as(i32, c.HETERIS_INVALID_ARGUMENT), bulkRange(&invocation, 2, 1));
    try std.testing.expectEqual(@as(i32, c.HETERIS_OUT_OF_BOUNDS), bulkRange(&invocation, 0, 4));
    try std.testing.expectEqual(@as(i32, c.HETERIS_OK), bulkRange(&invocation, 3, 3));
}
