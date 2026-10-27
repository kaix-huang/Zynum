// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
const std = @import("std");
pub const c = @import("heteris-provider-c");
const results = @import("results.zig");
const contracts = @import("contracts.zig");
const resources = @import("zynum-numerics-resources");
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
    pub fn acquire(self: *const Binding, comptime T: type, count: usize, report: *CallReport) ExecutionError!resources.ProviderScratchLease(T) {
        if (@sizeOf(T) == 0) @compileError("provider scratch requires nonzero-sized elements");
        const bytes = std.math.mul(usize, count, @sizeOf(T)) catch return error.ExtentOverflow;
        const bytes64 = std.math.cast(u64, bytes) orelse return error.ExtentOverflow;
        _ = std.math.add(usize, bytes, @alignOf(T) - 1) catch return error.ExtentOverflow;
        report.provider.stage = .scratch_allocate;
        var scratch: c.heteris_rp_scratch_v1 = std.mem.zeroes(c.heteris_rp_scratch_v1);
        const code = self.api.scratch_allocate.?(self.token, bytes64, @alignOf(T), &scratch);
        report.provider.returned_status = code;
        if (code != c.HETERIS_OK) return mapStatus(code);
        const charged = std.math.add(u64, bytes64, @alignOf(T) - 1) catch failStop(report, "scratch charge overflow");
        if (scratch.abi_version != 1 or scratch.struct_size < @sizeOf(@TypeOf(scratch)) or !tokenValid(scratch.lease, self.api.authority, c.HETERIS_RP_SCRATCH) or !sameIdentity(scratch.memory_domain, self.caps.memory_domain) or scratch.data == null or scratch.logical_bytes != bytes64 or scratch.alignment != @alignOf(T) or scratch.accounted_bytes != charged) failStop(report, "malformed scratch ownership descriptor");
        const address = @intFromPtr(scratch.data);
        if (address % @alignOf(T) != 0) failStop(report, "misaligned scratch backing");
        _ = std.math.add(usize, address, bytes) catch failStop(report, "scratch address extent overflow");
        return resources.ProviderScratchLease(T).initValidated(scratch, self.api.scratch_release.?, count);
    }
};
